#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.evaluate_existing_hybrid_retrieval import (
    QuerySpec,
    annotate_results,
    compute_keyword_hits,
    evaluate_chunk_against_expectation,
    find_target_chunks,
    load_chunks,
    load_query_specs,
    normalize_text,
    query_mentions_expected_doc,
    text_contains_keyword,
    to_rank_map,
    truncate_text,
)
from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.schemas import Citation, QueryAnalysis, QueryFilters, RAGResponse, RetrievedChunk


CONFIG_SCAN_PATHS = [
    ROOT / "src/synbio_rag/domain/config.py",
    ROOT / "src/synbio_rag/application/rerank_service.py",
    ROOT / "scripts/evaluation/evaluate_existing_hybrid_retrieval.py",
    ROOT / "scripts/evaluation/evaluate_guarded_reranker.py",
    ROOT / "tests/test_guarded_reranker.py",
]
REFUSAL_PATTERNS = (
    "证据不足",
    "无法可靠作答",
    "无法基于文库可靠回答",
    "没有提供足够可引用证据",
    "evidence is insufficient",
)
DNA_SEQ_RE = re.compile(r"\b[ACGT]{12,}\b", flags=re.IGNORECASE)
DOC_ID_RE = re.compile(r"\bdoc[_-]?\d{4}\b", flags=re.IGNORECASE)


@dataclass
class ConfigAuditResult:
    rerank_mode_vars: list[str]
    rank1_guard_vars: list[str]
    typo_vars: list[str]
    fixed_vars: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small-sample end-to-end generation evaluation through the formal BIORAG pipeline.")
    parser.add_argument("--query_spec", required=True)
    parser.add_argument("--chunks_jsonl", required=True)
    parser.add_argument("--collection_name", required=True)
    parser.add_argument("--milvus_uri", required=True)
    parser.add_argument("--embedding_model_path", required=True)
    parser.add_argument("--reranker_model_path", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--candidate_k", type=int, default=50)
    parser.add_argument("--rerank_candidate_k", type=int, default=10)
    parser.add_argument("--embedding_max_length", type=int, default=2048)
    parser.add_argument("--expected_guarded_mode", default="guarded_rank1")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def scan_config_variable_names(paths: list[Path] | None = None) -> ConfigAuditResult:
    targets = paths or CONFIG_SCAN_PATHS
    found: set[str] = set()
    for path in targets:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        found.update(re.findall(r"[A-Z][A-Z0-9_]{5,}", text))
    rerank_mode_vars = sorted(var for var in found if "RERANK_MODE" in var)
    rank1_guard_vars = sorted(
        var
        for var in found
        if "RANK1" in var or "MAX_SCORE_GAP" in var or "MIN_COMPLETENESS" in var
    )
    typo_vars = sorted(var for var in found if "RETRIEVALANK1" in var or "RETRIEVAL_RANK1_" in var)
    fixed_vars: list[str] = []
    if "RETRIEVAL_GUARDED_RANK1_MAX_SCORE_GAP" in found:
        fixed_vars.append("RETRIEVAL_GUARDED_RANK1_MAX_SCORE_GAP")
    if "RETRIEVAL_GUARDED_RANK1_MIN_COMPLETENESS_GAIN" in found:
        fixed_vars.append("RETRIEVAL_GUARDED_RANK1_MIN_COMPLETENESS_GAIN")
    return ConfigAuditResult(
        rerank_mode_vars=rerank_mode_vars,
        rank1_guard_vars=rank1_guard_vars,
        typo_vars=typo_vars,
        fixed_vars=fixed_vars,
    )


def resolve_effective_rerank_mode() -> str:
    return Settings.from_env().retrieval.rerank_mode


def assert_expected_guarded_mode(expected_mode: str) -> str:
    effective = resolve_effective_rerank_mode()
    if expected_mode not in {"guarded", "guarded_rank1"}:
        raise AssertionError(f"unsupported expected guarded mode: {expected_mode}")
    if effective not in {"guarded", "guarded_rank1"}:
        raise AssertionError(
            f"expected guarded evaluation mode, but effective rerank mode is {effective!r}"
        )
    if effective != expected_mode:
        raise AssertionError(
            f"expected rerank mode {expected_mode!r}, but effective rerank mode is {effective!r}"
        )
    return effective


def apply_runtime_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    settings.retrieval.milvus_uri = str(Path(args.milvus_uri).resolve())
    settings.retrieval.collection_name = args.collection_name
    settings.kb.chunk_jsonl = str(Path(args.chunks_jsonl).resolve())
    settings.kb.chunk_dir = str(Path(args.chunks_jsonl).resolve().parent)
    settings.kb.embedding_model_path = str(Path(args.embedding_model_path).resolve())
    settings.kb.embedding_max_length = args.embedding_max_length
    settings.reranker.model_path = str(Path(args.reranker_model_path).resolve())
    settings.retrieval.search_limit = args.candidate_k
    settings.retrieval.dense_limit = args.candidate_k
    settings.retrieval.bm25_limit = args.candidate_k
    settings.retrieval.rerank_top_k = max(args.top_k, args.rerank_candidate_k)
    settings.retrieval.final_top_k = args.top_k
    settings.retrieval.hybrid_enabled = True
    settings.retrieval.bm25_enabled = True
    return settings


def serialize_citations(citations: list[Citation]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in citations:
        rows.append(
            {
                "chunk_id": item.chunk_id,
                "doc_id": item.doc_id,
                "title": item.title,
                "source_file": item.source_file,
                "section": item.section,
                "page_start": item.page_start,
                "page_end": item.page_end,
                "score": float(item.score or 0.0),
                "quote": item.quote,
            }
        )
    return rows


def run_formal_pipeline(
    pipeline: SynBioRAGPipeline,
    question: str,
    *,
    rerank_mode: str,
    filters: QueryFilters | None,
) -> dict[str, Any]:
    start_settings_mode = pipeline.settings.retrieval.rerank_mode
    analysis: QueryAnalysis = pipeline.router.analyze(question)
    retrieved, retrieval_debug = pipeline._search_with_filter_fallback(
        question=question,
        analysis=analysis,
        filters=filters,
    )
    reranked = pipeline.reranker.rerank(
        question,
        retrieved,
        top_k=analysis.rerank_top_k,
        analysis=analysis,
        mode=rerank_mode,
    )
    seed_chunks = reranked[: pipeline.settings.retrieval.final_top_k]
    final_chunks = pipeline.neighbor_expander.expand(seed_chunks)
    context = pipeline.context_builder.build(question, final_chunks, history=None, intent=analysis.intent)
    evidence_quality = pipeline.generator.assess_evidence(question, final_chunks, analysis=analysis)
    answer = pipeline.generator.generate(
        question,
        context,
        final_chunks,
        analysis=analysis,
        history=None,
        assessment=evidence_quality,
    )
    citations = pipeline.generator.build_citations(final_chunks, evidence_quality)
    answer = pipeline.generator.validate_generated_answer(answer, citations, evidence_quality)
    response = RAGResponse(
        answer=answer,
        confidence=pipeline.confidence_scorer.score(final_chunks),
        route=analysis.intent,
        citations=citations,
        used_external_tool=False,
        tool_name=None,
        tool_result=None,
        debug={
            "retrieved_count": len(retrieved),
            "reranked_count": len(reranked),
            "seed_context_count": len(seed_chunks),
            "final_context_count": len(final_chunks),
            "hybrid_enabled": pipeline.settings.retrieval.hybrid_enabled,
            "bm25_enabled": pipeline.settings.retrieval.bm25_enabled,
            "effective_rerank_mode": rerank_mode,
            "settings_rerank_mode": start_settings_mode,
            "retrieval_hits": getattr(pipeline.retriever, "last_debug", {}),
            "rerank_hits": getattr(pipeline.reranker, "last_debug", {}),
            "neighbor_expansion": getattr(pipeline.neighbor_expander, "last_debug", {}),
            "filter_strategy": retrieval_debug,
            "evidence_quality": evidence_quality.__dict__,
        },
    )
    return {
        "analysis": analysis,
        "retrieved": retrieved,
        "reranked": reranked,
        "seed_chunks": seed_chunks,
        "final_chunks": final_chunks,
        "response": response,
    }


def is_refusal(answer: str) -> bool:
    lowered = (answer or "").strip().lower()
    return any(pattern.lower() in lowered for pattern in REFUSAL_PATTERNS)


def citation_matches_target(spec: QuerySpec, citation: dict[str, Any]) -> bool:
    if spec.expected_doc_id and citation.get("doc_id") != spec.expected_doc_id:
        return False
    quote = citation.get("quote", "")
    matches = sum(1 for keyword in spec.expected_keywords if text_contains_keyword(quote, keyword))
    if spec.category == "figure":
        return matches >= 1 and "figure_caption" in normalize_text(citation.get("quote", "") + " " + citation.get("section", ""))
    if spec.category == "table":
        return matches >= max(1, len(spec.expected_keywords) // 2)
    return matches >= max(1, min(2, len(spec.expected_keywords)))


def answer_expected_hit_count(spec: QuerySpec, answer: str) -> tuple[int, list[str]]:
    required = ANSWER_KEYWORDS_BY_QUERY.get(spec.id, spec.expected_keywords)
    hits, matched = compute_keyword_hits(answer, required)
    return hits, matched


ANSWER_KEYWORDS_BY_QUERY: dict[str, list[str]] = {
    "q_table3_man8": ["Man8", "PpFWK3HRP", "57.3"],
    "q_table4_vmax": ["KM_ABTS", "Vmax", "PpFWK3HRP", "2.03", "2.46"],
    "q_table5_primer": ["OCH1-5int-fw1", "OCH1-5int-rv1"],
    "q_table6_strains": ["strain name", "PpMutS", "PpFWK3", "PpMutSHRP", "PpFWK3HRP"],
    "q_doc0001_materials": ["HMOs", "Glycom A/S"],
    "q_doc0001_glycom": ["Glycom", "2′-FL", "LNT", "LNnT"],
    "q_doc0045_fig1": ["Figure 1"],
    "q_doc0001_fig4": ["Fig. 4", "Microbial composition changes", "top 20 species"],
    "q_doc0001_2fl": ["2′-FL", "Bifidobacterium"],
}


def answer_meets_query_expectation(spec: QuerySpec, answer: str) -> tuple[bool, bool, str]:
    answer_norm = normalize_text(answer)
    hits, _ = answer_expected_hit_count(spec, answer)
    if spec.id == "q_table2_parameters":
        refusal = is_refusal(answer)
        return refusal, refusal, "expected_refusal_pass" if refusal else "refusal_failure"
    if spec.id == "q_table5_primer":
        has_names = all(text_contains_keyword(answer, kw) for kw in ANSWER_KEYWORDS_BY_QUERY[spec.id])
        has_sequence = len(DNA_SEQ_RE.findall(answer or "")) >= 1
        if has_names and has_sequence:
            return True, True, "pass"
        if has_names:
            return True, False, "partial"
        return False, False, "fail"
    if spec.category == "table":
        required = ANSWER_KEYWORDS_BY_QUERY.get(spec.id, spec.expected_keywords)
        pass_threshold = max(2, len(required) - 1)
        partial_threshold = max(1, len(required) // 2)
        if hits >= pass_threshold:
            return True, True, "pass"
        if hits >= partial_threshold:
            return True, False, "partial"
        return False, False, "fail"
    if spec.category == "figure":
        if is_refusal(answer):
            return False, False, "fail"
        if len(answer_norm) >= 24:
            return True, hits >= 1, "pass" if hits >= 1 else "partial"
        return False, False, "fail"
    required = ANSWER_KEYWORDS_BY_QUERY.get(spec.id, spec.expected_keywords)
    pass_threshold = max(2, len(required) - 1)
    partial_threshold = max(1, min(2, len(required)))
    if hits >= pass_threshold:
        return True, True, "pass"
    if hits >= partial_threshold:
        return True, False, "partial"
    return False, False, "fail"


def detect_hallucination(spec: QuerySpec, answer: str, citations: list[dict[str, Any]]) -> bool:
    if spec.id == "q_table2_parameters":
        if is_refusal(answer):
            return False
        return bool(re.search(r"\d", answer))
    evidence_text = " ".join((citation.get("quote") or "") for citation in citations)
    if not evidence_text:
        return False
    answer_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", answer))
    evidence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", evidence_text))
    extra_numbers = answer_numbers - evidence_numbers
    return len(extra_numbers) > 1


def grade_mode(
    spec: QuerySpec,
    reranked_rows: list[dict[str, Any]],
    response: RAGResponse,
) -> dict[str, Any]:
    retrieval_hit = any(row.get("target_match") for row in reranked_rows)
    rank1_row = reranked_rows[0] if reranked_rows else None
    rank1_complete = bool(rank1_row and rank1_row.get("target_match"))
    citations = serialize_citations(response.citations)
    citation_valid = any(citation_matches_target(spec, citation) for citation in citations)
    answer = response.answer or ""
    answer_supported, answer_contains_expected, answer_grade = answer_meets_query_expectation(spec, answer)
    hallucination = detect_hallucination(spec, answer, citations)
    support_pack = response.debug.get("evidence_quality", {}).get("support_pack", [])
    prompt_support_context = response.debug.get("evidence_quality", {}).get("prompt_support_context", "")
    target_chunk_ids = [row["chunk_id"] for row in reranked_rows if row.get("target_match")]
    support_chunk_ids = [str(item.get("chunk_id") or "") for item in support_pack]
    support_pack_drop = bool(target_chunk_ids) and not any(chunk_id in support_chunk_ids for chunk_id in target_chunk_ids)
    prompt_context_drop = bool(target_chunk_ids) and not any(chunk_id and chunk_id in prompt_support_context for chunk_id in target_chunk_ids)

    if spec.id == "q_table2_parameters":
        if answer_grade == "expected_refusal_pass" and not hallucination:
            grade = "expected_refusal_pass"
            failure_reason = "extraction_gap"
        else:
            grade = "fail"
            failure_reason = "refusal_failure"
    elif not retrieval_hit:
        grade = "fail"
        failure_reason = "retrieval_failure"
    elif support_pack_drop:
        grade = "fail"
        failure_reason = "support_pack_drop"
    elif prompt_context_drop:
        grade = "fail"
        failure_reason = "prompt_context_drop"
    elif not rank1_complete and spec.category in {"table", "figure"} and not answer_supported:
        grade = "partial" if citation_valid else "fail"
        failure_reason = "rank_failure"
    elif not citation_valid:
        grade = "partial" if answer_supported else "fail"
        failure_reason = "citation_failure"
    elif not answer_supported:
        grade = "fail"
        failure_reason = "generation_failure"
    elif hallucination:
        grade = "partial"
        failure_reason = "generation_failure"
    else:
        grade = answer_grade
        failure_reason = "ok"

    return {
        "retrieval_hit": retrieval_hit,
        "rank1_complete": rank1_complete,
        "citation_valid": citation_valid,
        "answer_grounded": citation_valid and not hallucination,
        "answer_contains_expected": answer_contains_expected,
        "hallucination": hallucination,
        "refusal_quality": answer_grade == "expected_refusal_pass",
        "support_pack_drop": support_pack_drop,
        "prompt_context_drop": prompt_context_drop,
        "grade": grade,
        "failure_reason": failure_reason,
        "answer_expected_hits": answer_expected_hit_count(spec, answer)[0],
        "answer_matched_keywords": answer_expected_hit_count(spec, answer)[1],
    }


def build_mode_result(
    spec: QuerySpec,
    result: dict[str, Any],
    chunk_map: dict[str, dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    reranked = result["reranked"]
    dense_rank_map = to_rank_map(result["retrieved"])
    reranked_rows = annotate_results(reranked[:top_k], spec, chunk_map, dense_rank_map, {})
    support_pack = result["response"].debug.get("evidence_quality", {}).get("support_pack", [])
    prompt_support_context = result["response"].debug.get("evidence_quality", {}).get("prompt_support_context", "")
    target_chunk_ids = [row["chunk_id"] for row in reranked_rows if row.get("target_match")]
    support_chunk_ids = [str(item.get("chunk_id") or "") for item in support_pack]
    support_pack_drop = bool(target_chunk_ids) and not any(chunk_id in support_chunk_ids for chunk_id in target_chunk_ids)
    prompt_context_drop = bool(target_chunk_ids) and not any(chunk_id and chunk_id in prompt_support_context for chunk_id in target_chunk_ids)
    graded = grade_mode(spec, reranked_rows, result["response"])
    return {
        "rerank_mode": result["response"].debug.get("effective_rerank_mode"),
        "route": str(result["response"].route),
        "answer": result["response"].answer,
        "citations": serialize_citations(result["response"].citations),
        "retrieved_topk": reranked_rows,
        "rank1_evidence": reranked_rows[0] if reranked_rows else None,
        "support_pack": support_pack,
        "prompt_support_context": prompt_support_context,
        "support_pack_drop": support_pack_drop,
        "prompt_context_drop": prompt_context_drop,
        "debug": result["response"].debug,
        **graded,
    }


def run_text(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()


def report_query_block(spec: QuerySpec, mode_rows: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        f"### {spec.id}",
        f"- Query: `{spec.query}`",
        f"- Category: `{spec.category}`",
        f"- expected_doc_id: `{spec.expected_doc_id}`",
        f"- expected_keywords: `{spec.expected_keywords}`",
        f"- expected_block_types: `{spec.expected_block_types}`",
        f"- Excluded from primary metrics: `{spec.exclude_from_primary_metrics}`",
    ]
    for mode_name, payload in mode_rows.items():
        lines.append(
            f"- {mode_name}: grade=`{payload['grade']}` retrieval_hit=`{payload['retrieval_hit']}` "
            f"rank1_complete=`{payload['rank1_complete']}` citation_valid=`{payload['citation_valid']}` "
            f"answer_grounded=`{payload['answer_grounded']}` answer_contains_expected=`{payload['answer_contains_expected']}` "
            f"hallucination=`{payload['hallucination']}` support_pack_drop=`{payload['support_pack_drop']}` "
            f"prompt_context_drop=`{payload['prompt_context_drop']}` failure_reason=`{payload['failure_reason']}` rerank_mode=`{payload['rerank_mode']}`"
        )
        rank1 = payload["rank1_evidence"]
        if rank1:
            lines.append(
                "  - "
                f"rank1 chunk_id=`{rank1['chunk_id']}` doc_id=`{rank1['doc_id']}` section=`{rank1['section']}` "
                f"block_types=`{rank1['block_types']}` expected_doc_hit=`{rank1['expected_doc_hit']}` "
                f"expected_keywords_hit=`{rank1['full_keyword_hit']}` expected_block_types_hit=`{rank1['expected_block_hit']}` "
                f"guarded_completeness=`{rank1['guarded_completeness_score']:.3f}` text=`{truncate_text(rank1['text'])}`"
            )
        lines.append(f"  - Answer: `{payload['answer']}`")
        lines.append(f"  - Answer matched keywords: `{payload['answer_matched_keywords']}`")
        prompt_context = payload.get("prompt_support_context") or ""
        lines.append(
            f"  - Prompt context contains expected doc marker: `{spec.expected_doc_id in prompt_context if spec.expected_doc_id else 'n/a'}` "
            f"contains expected keywords: `{any(text_contains_keyword(prompt_context, keyword) for keyword in spec.expected_keywords)}`"
        )
        if payload["citations"]:
            lines.append("  - Citations:")
            for citation in payload["citations"]:
                lines.append(
                    "    - "
                    f"chunk_id=`{citation['chunk_id']}` doc_id=`{citation['doc_id']}` section=`{citation['section']}` "
                    f"pages=`{citation['page_start']}-{citation['page_end']}` quote=`{truncate_text(citation['quote'], 260)}`"
                )
        else:
            lines.append("  - Citations: `[]`")
        lines.append("  - Retrieved top5:")
        for row in payload["retrieved_topk"]:
            lines.append(
                "    - "
                f"rank=`{row['final_rank']}` chunk_id=`{row['chunk_id']}` doc_id=`{row['doc_id']}` section=`{row['section']}` "
                f"block_types=`{row['block_types']}` expected_doc_hit=`{row['expected_doc_hit']}` "
                f"expected_keywords_hit=`{row['full_keyword_hit']}` expected_block_types_hit=`{row['expected_block_hit']}` "
                f"matched_keywords=`{row['matched_keywords']}` text=`{truncate_text(row['text'], 220)}`"
            )
        lines.append("  - Support pack:")
        for item in payload.get("support_pack", [])[:5]:
            lines.append(
                "    - "
                f"chunk_id=`{item.get('chunk_id')}` doc_id=`{item.get('doc_id')}` section=`{item.get('section')}` "
                f"block_types=`{item.get('block_types')}` support_type=`{item.get('support_type')}` "
                f"support_score=`{item.get('support_score')}` quote=`{truncate_text(str(item.get('quote') or ''), 220)}`"
            )
    return lines


def summarize_mode(query_results: list[dict[str, Any]], mode_name: str) -> tuple[int, int, dict[str, int]]:
    primary = [item for item in query_results if not item["spec"].exclude_from_primary_metrics]
    passed = sum(1 for item in primary if item["modes"][mode_name]["grade"] == "pass")
    denominator = len(primary)
    counts = Counter(item["modes"][mode_name]["grade"] for item in primary)
    return passed, denominator, dict(counts)


def build_report(
    args: argparse.Namespace,
    settings: Settings,
    config_audit: ConfigAuditResult,
    query_results: list[dict[str, Any]],
) -> str:
    branch = run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    commit = run_text(["git", "rev-parse", "HEAD"])
    guarded_pass, denominator, guarded_counts = summarize_mode(query_results, "hybrid_guarded_rank1")
    plain_pass, _, plain_counts = summarize_mode(query_results, "baseline_plain")
    doc_pass, _, doc_counts = summarize_mode(query_results, "hybrid_guarded_rank1_doc_routed")
    table_queries = [item for item in query_results if item["spec"].category == "table" and not item["spec"].exclude_from_primary_metrics]
    figure_queries = [item for item in query_results if item["spec"].category == "figure"]
    body_queries = [item for item in query_results if item["spec"].category == "body"]
    table_pass = sum(1 for item in table_queries if item["modes"]["hybrid_guarded_rank1"]["grade"] == "pass")
    figure_pass = sum(1 for item in figure_queries if item["modes"]["hybrid_guarded_rank1"]["grade"] == "pass")
    body_pass = sum(1 for item in body_queries if item["modes"]["hybrid_guarded_rank1"]["grade"] == "pass")
    citation_valid_rate = sum(1 for item in query_results if item["modes"]["hybrid_guarded_rank1"]["citation_valid"]) / max(len(query_results), 1)
    support_pack_drop_count = sum(1 for item in query_results if item["modes"]["hybrid_guarded_rank1"]["support_pack_drop"])
    prompt_context_drop_count = sum(1 for item in query_results if item["modes"]["hybrid_guarded_rank1"]["prompt_context_drop"])
    generation_fail_count = sum(1 for item in query_results if item["modes"]["hybrid_guarded_rank1"]["failure_reason"] == "generation_failure")
    citation_fail_count = sum(1 for item in query_results if item["modes"]["hybrid_guarded_rank1"]["failure_reason"] == "citation_failure")
    config_values = {
        "BIORAG_RERANK_MODE": os.getenv("BIORAG_RERANK_MODE", ""),
        "RETRIEVAL_RERANK_MODE": os.getenv("RETRIEVAL_RERANK_MODE", ""),
        "RETRIEVAL_GUARDED_RANK1_MIN_COMPLETENESS_GAIN": os.getenv("RETRIEVAL_GUARDED_RANK1_MIN_COMPLETENESS_GAIN", str(settings.retrieval.guarded_rank1_min_completeness_gain)),
        "RETRIEVAL_GUARDED_RANK1_MAX_SCORE_GAP": os.getenv("RETRIEVAL_GUARDED_RANK1_MAX_SCORE_GAP", str(settings.retrieval.guarded_rank1_max_score_gap)),
    }
    lines = [
        "# E2E Guarded Rank1 Eval",
        "",
        "## Run Metadata",
        f"- Branch: `{branch}`",
        f"- Commit: `{commit}`",
        f"- Collection: `{args.collection_name}`",
        f"- Milvus URI: `{Path(args.milvus_uri).resolve()}`",
        f"- Query spec: `{Path(args.query_spec).resolve()}`",
        f"- Chunks JSONL: `{Path(args.chunks_jsonl).resolve()}`",
        f"- Embedding model: `{Path(args.embedding_model_path).resolve()}`",
        f"- Reranker model: `{Path(args.reranker_model_path).resolve()}`",
        f"- Generation provider/model: `{settings.llm.provider}` / `{settings.llm.model_name}`",
        "",
        "## 配置变量名检查",
        f"- Rerank mode variables found: `{config_audit.rerank_mode_vars}`",
        f"- Rank1 guard variables found: `{config_audit.rank1_guard_vars}`",
        f"- Typo-like variables found: `{config_audit.typo_vars}`",
        f"- Fixed/confirmed canonical variables: `{config_audit.fixed_vars}`",
        f"- Actual evaluation variables: `{config_values}`",
        f"- Effective rerank mode required by this run: `{args.expected_guarded_mode}`",
        f"- Rank1 guard enabled in this run: `{args.expected_guarded_mode == 'guarded_rank1'}`",
        f"- Doc-routed comparison enabled: `True`",
        "",
        "## Summary Metrics",
        f"- baseline_plain pass/denominator: `{plain_pass}/{denominator}` summary=`{plain_counts}`",
        f"- hybrid_guarded_rank1 pass/denominator: `{guarded_pass}/{denominator}` summary=`{guarded_counts}`",
        f"- hybrid_guarded_rank1_doc_routed pass/denominator: `{doc_pass}/{denominator}` summary=`{doc_counts}`",
        f"- citation_valid_rate (guarded_rank1): `{citation_valid_rate:.3f}`",
        f"- table_answer_pass_rate: `{table_pass}/{len(table_queries)}`",
        f"- figure_answer_pass_rate: `{figure_pass}/{len(figure_queries)}`",
        f"- body_answer_pass_rate: `{body_pass}/{len(body_queries)}`",
        f"- support_pack_drop_count (guarded_rank1): `{support_pack_drop_count}`",
        f"- prompt_context_drop_count (guarded_rank1): `{prompt_context_drop_count}`",
        f"- generation_failure_count (guarded_rank1): `{generation_fail_count}`",
        f"- citation_failure_count (guarded_rank1): `{citation_fail_count}`",
        "",
        "## Failure Attribution Summary",
        f"- guarded_rank1 failure reasons: `{dict(Counter(item['modes']['hybrid_guarded_rank1']['failure_reason'] for item in query_results))}`",
        f"- plain failure reasons: `{dict(Counter(item['modes']['baseline_plain']['failure_reason'] for item in query_results))}`",
        "",
        "## Per-Query Results",
    ]
    for item in query_results:
        lines.extend(report_query_block(item["spec"], item["modes"]))
        lines.append("")
    lines.extend(
        [
            "## Conclusions",
            f"- guarded_rank1 answer pass rate (excluding extraction gap): `{guarded_pass}/{denominator}`",
            f"- plain answer pass rate (excluding extraction gap): `{plain_pass}/{denominator}`",
            f"- q_table2_parameters refusal quality: `{next(item for item in query_results if item['spec'].id == 'q_table2_parameters')['modes']['hybrid_guarded_rank1']['grade']}`",
            f"- retrieval-hit but generation-fail queries (guarded_rank1): `{[item['spec'].id for item in query_results if item['modes']['hybrid_guarded_rank1']['retrieval_hit'] and item['modes']['hybrid_guarded_rank1']['failure_reason'] == 'generation_failure']}`",
            f"- citation-fail queries (guarded_rank1): `{[item['spec'].id for item in query_results if item['modes']['hybrid_guarded_rank1']['failure_reason'] == 'citation_failure']}`",
            "- Recommendation on larger-sample E2E eval: proceed only if guarded_rank1 stays explicitly enabled; do not run larger E2E with default plain reranking.",
            "- Recommendation on Table 2: keep as a separate extraction fix before expecting answerable behavior beyond refusal.",
            "- Recommendation on Milvus BM25 A/B: still defer it. Current E2E gate is answer grounding, not sparse backend parity.",
            "- Recommendation on false-heading cleanup: still defer it unless new misses can be traced to noisy section metadata.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config_audit = scan_config_variable_names()
    if config_audit.typo_vars:
        raise SystemExit(f"unexpected typo-like config vars found: {config_audit.typo_vars}")

    effective_mode = assert_expected_guarded_mode(args.expected_guarded_mode)
    if effective_mode != args.expected_guarded_mode:
        raise SystemExit(
            f"guarded evaluation requested {args.expected_guarded_mode!r}, but effective mode is {effective_mode!r}"
        )

    specs = load_query_specs(args.query_spec)
    _chunks, chunk_map = load_chunks(args.chunks_jsonl)
    settings = apply_runtime_overrides(Settings.from_env(), args)
    pipeline = SynBioRAGPipeline(settings)

    query_results: list[dict[str, Any]] = []
    for spec in specs:
        guarded = run_formal_pipeline(
            pipeline,
            spec.query,
            rerank_mode="guarded_rank1",
            filters=None,
        )
        plain = run_formal_pipeline(
            pipeline,
            spec.query,
            rerank_mode="plain",
            filters=None,
        )
        routed_filters = QueryFilters(doc_ids=[spec.expected_doc_id]) if query_mentions_expected_doc(spec) and spec.expected_doc_id else None
        doc_routed = run_formal_pipeline(
            pipeline,
            spec.query,
            rerank_mode="guarded_rank1",
            filters=routed_filters,
        )
        query_results.append(
            {
                "spec": spec,
                "target_exists": find_target_chunks(spec, list(chunk_map.values()))["target_exists_in_chunks"],
                "modes": {
                    "baseline_plain": build_mode_result(spec, plain, chunk_map, args.top_k),
                    "hybrid_guarded_rank1": build_mode_result(spec, guarded, chunk_map, args.top_k),
                    "hybrid_guarded_rank1_doc_routed": build_mode_result(spec, doc_routed, chunk_map, args.top_k),
                },
            }
        )

    report = build_report(args, settings, config_audit, query_results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
