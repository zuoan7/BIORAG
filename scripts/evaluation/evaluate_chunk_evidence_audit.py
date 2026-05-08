#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag.application.rerank_service import QwenReranker
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.router import QueryRouter
from src.synbio_rag.domain.schemas import QueryAnalysis, QueryIntent, RetrievedChunk
from src.synbio_rag.infrastructure.embedding.bge import BGEM3Embedder
from src.synbio_rag.infrastructure.vectorstores.bm25 import BM25Retriever, tokenize_query
from src.synbio_rag.infrastructure.vectorstores.hybrid import HybridRetriever
from src.synbio_rag.infrastructure.vectorstores.milvus import MilvusRetriever


DEFAULT_DATASET = Path("data/eval/datasets/enterprise_ragas_smoke100.json")
DEFAULT_PHASE16R_METRICS = Path("results/phase16r_retrieval_only_baseline/retrieval_only_metrics.json")
DEFAULT_OUTPUT_DIR = Path("results/phase16r2_chunk_evidence_audit")
DEFAULT_REPORT_DIR = Path("reports/phase16r2_chunk_evidence_audit")

STOPWORDS_ZH = {
    "文库", "研究", "作者", "总结", "说明", "比较", "主要", "关键", "结果", "结论", "哪些", "什么", "如何",
    "证据", "问题", "论文", "文献", "指出", "回答", "应指", "是否", "关于", "相关", "作用", "核心",
}
STOPWORDS_EN = {
    "the", "and", "for", "with", "from", "into", "that", "this", "study", "paper", "result", "results",
    "conclusion", "conclusions", "main", "key", "what", "which", "how", "why", "using", "used",
}
DOMAIN_ZH_TERMS = {
    "骨质疏松", "巨噬细胞", "糖基化", "磷酸化", "启动子", "转运蛋白", "分泌", "毕赤酵母", "大肠杆菌",
    "人乳寡糖", "唾液酸", "膜转运", "芳香化合物", "生物传感器", "代谢工程", "岩藻糖基乳糖",
    "岩藻糖基化", "乳糖", "发酵", "乙酸", "乳酸", "通路", "酶", "蛋白", "菌株", "产量", "滴度",
    "生物修复", "木质素", "塑料降解", "外排泵", "跨膜", "底物", "毒性", "高值化",
}
DOMAIN_EN_TERMS = {
    "2'-fl", "2-fl", "2-fucosyllactose", "3-fl", "3'-sl", "6'-sl", "6-sialyllactose",
    "6'-sialyllactose", "neu5ac", "hmo", "hmos", "opn", "fam20a", "fam20c", "kex2", "hac1",
    "m1", "ph", "e. coli", "pichia", "komagataella", "bifidobacterium", "mfs", "abc", "trap",
    "fadl", "rnd", "nanr", "mkate2", "crispr", "cas9", "glycosylation", "phosphorylation",
}
NOISE_SECTION_RE = re.compile(r"\b(references?|bibliography|acknowledg(e)?ments?|funding|conflicts?)\b", re.I)
TOKEN_RE = re.compile(
    r"(?:[A-Za-z]\.\s*)?[A-Za-z][A-Za-z0-9'_.+\-]*[A-Za-z0-9]|\d+(?:\.\d+)?(?:\s?%|[-']?[A-Za-z]+)?|[\u4e00-\u9fff]{2,8}"
)


KEYWORD_PREVIEW_FIELDNAMES = [
    "sample_id", "question", "reference_answer", "expected_doc_ids", "expected_sections",
    "extracted_question_keywords", "extracted_answer_keywords", "extracted_domain_entities",
    "normalized_keyword_groups", "keyword_count", "notes",
]
PER_SAMPLE_FIELDNAMES = [
    "sample_id", "question", "expected_doc_ids", "expected_source_files", "expected_sections",
    "negative_query", "skipped_doc_hit", "reference_answer",
    "dense_doc_best_rank", "bm25_doc_best_rank", "hybrid_doc_best_rank", "rerank_doc_best_rank",
    "dense_best_answerable_chunk_rank", "bm25_best_answerable_chunk_rank",
    "hybrid_best_answerable_chunk_rank", "rerank_best_answerable_chunk_rank",
    "dense_best_strong_evidence_rank", "bm25_best_strong_evidence_rank",
    "hybrid_best_strong_evidence_rank", "rerank_best_strong_evidence_rank",
    "dense_answerable_hit_at_10", "dense_answerable_hit_at_20", "dense_answerable_hit_at_40",
    "bm25_answerable_hit_at_10", "bm25_answerable_hit_at_20", "bm25_answerable_hit_at_40",
    "hybrid_answerable_hit_at_10", "hybrid_answerable_hit_at_20", "hybrid_answerable_hit_at_40",
    "rerank_answerable_hit_at_5", "rerank_answerable_hit_at_10", "rerank_answerable_hit_at_20",
    "rerank_strong_evidence_hit_at_5", "rerank_strong_evidence_hit_at_10", "rerank_strong_evidence_hit_at_20",
    "doc_hit_but_evidence_miss", "evidence_found_but_reranker_suppressed",
    "evidence_found_in_hybrid_but_not_rerank", "evidence_retrieval_status", "recommended_next_action",
]
RERANK_TRACE_FIELDNAMES = [
    "sample_id", "rerank_rank", "chunk_id", "doc_id", "source_file", "section", "section_path",
    "rerank_score", "dense_score_if_available", "bm25_score_if_available", "hybrid_score_if_available",
    "expected_doc_chunk", "expected_section_chunk", "answer_keyword_overlap", "question_keyword_overlap",
    "matched_answer_keywords", "matched_question_keywords", "evidence_score_simple", "answerable_chunk_weak",
    "strong_evidence_chunk", "section_noise_flag", "text_preview",
]
MISS_FIELDNAMES = [
    "sample_id", "question", "expected_doc_ids", "rerank_doc_best_rank",
    "expected_doc_chunks_in_rerank_top10", "expected_doc_chunk_sections",
    "extracted_answer_keywords", "extracted_question_keywords", "why_evidence_miss", "recommended_next_action",
]


@dataclass(frozen=True)
class KeywordProfile:
    question_keywords: list[str]
    answer_keywords: list[str]
    domain_entities: list[str]
    normalized_groups: dict[str, list[str]]


@dataclass(frozen=True)
class EvidenceEval:
    expected_doc_chunk: bool
    expected_section_chunk: bool
    answer_keyword_overlap: int
    question_keyword_overlap: int
    matched_answer_keywords: list[str]
    matched_question_keywords: list[str]
    evidence_score_simple: int
    answerable_chunk_weak: bool
    strong_evidence_chunk: bool
    section_noise_flag: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 16R-2 chunk/evidence-level retrieval audit.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--phase16r-metrics", default=str(DEFAULT_PHASE16R_METRICS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dense-top-k", type=int, default=40)
    parser.add_argument("--bm25-top-k", type=int, default=40)
    parser.add_argument("--hybrid-top-k", type=int, default=40)
    parser.add_argument("--rerank-top-k", type=int, default=20)
    parser.add_argument("--command-used", default="")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_tests()
        return

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(dataset_path)
    if args.limit > 0:
        samples = samples[: args.limit]
    validate_dataset(samples)
    tokenizer_validation = validate_bm25_tokenizer(samples)
    phase16r_metrics = load_json(Path(args.phase16r_metrics))

    settings = Settings.from_env()
    settings.generation.version = "v2"
    settings.generation.v2_use_qwen_synthesis = False
    settings.generation.v2_enable_comparison_coverage = False
    settings.generation.v2_enable_neighbor_audit = False
    settings.generation.v2_enable_neighbor_promotion = False
    settings.generation.v2_include_neighbor_context_in_qwen = False
    settings.retrieval.parent_expansion_enabled = False
    settings.retrieval.bm25_enabled = True
    settings.retrieval.hybrid_enabled = True

    embedder = BGEM3Embedder(
        model_path=settings.kb.embedding_model_path,
        dim=settings.kb.embedding_dim,
        max_length=settings.kb.embedding_max_length,
    )
    router = QueryRouter(settings.retrieval)
    dense = MilvusRetriever(settings.retrieval, embedder)
    bm25 = BM25Retriever(settings.retrieval, settings.kb, milvus_client=dense.client)
    hybrid = HybridRetriever(settings.retrieval, dense, bm25)
    reranker = QwenReranker(
        api_base="",
        api_key="",
        model_name=settings.reranker.model_name,
        model_path=settings.reranker.model_path,
        service_url=settings.reranker.service_url,
        batch_size=settings.reranker.batch_size,
        use_fp16=settings.reranker.use_fp16,
        retrieval_config=settings.retrieval,
    )

    keyword_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    rerank_trace_rows: list[dict[str, Any]] = []
    doc_hit_miss_rows: list[dict[str, Any]] = []

    for index, sample in enumerate(samples, start=1):
        question = str(sample.get("question") or "")
        analysis = analysis_for_sample(router, sample, question)
        negative_query = bool(sample.get("negative_query"))
        skipped_doc_hit = negative_query and sample.get("should_require_doc_hit") is False
        profile = extract_keyword_profile(sample)
        keyword_rows.append(build_keyword_preview_row(sample, profile))

        dense_hits = dense.search(question, limit=max(args.dense_top_k, 40), filters=None)
        bm25_hits = bm25.search(question, limit=max(args.bm25_top_k, 40), filters=None)
        hybrid_hits = hybrid.search(question, limit=max(args.hybrid_top_k, 40), filters=None, analysis=analysis)
        reranked_hits = reranker.rerank(question, list(hybrid_hits), top_k=args.rerank_top_k, analysis=analysis)

        stage_evals = {
            "dense": evaluate_stage(sample, profile, dense_hits),
            "bm25": evaluate_stage(sample, profile, bm25_hits),
            "hybrid": evaluate_stage(sample, profile, hybrid_hits),
            "rerank": evaluate_stage(sample, profile, reranked_hits),
        }
        per_row = build_per_sample_row(sample, skipped_doc_hit, stage_evals)
        per_sample_rows.append(per_row)
        sample_rerank_trace_rows = build_rerank_trace_rows(sample, profile, reranked_hits)
        _RERANK_TRACE_CACHE[sample_id(sample)] = sample_rerank_trace_rows
        rerank_trace_rows.extend(sample_rerank_trace_rows)
        miss_row = build_doc_hit_but_evidence_miss_row(sample, profile, reranked_hits, stage_evals, per_row)
        if miss_row:
            doc_hit_miss_rows.append(miss_row)

        print(f"[{index}/{len(samples)}] {sample_id(sample)} {per_row['evidence_retrieval_status']}", flush=True)

    metrics = compute_metrics(samples, per_sample_rows, phase16r_metrics)
    metrics["bm25_tokenizer_validation"] = tokenizer_validation
    run_config = build_run_config(args, settings, dataset_path, args.command_used or " ".join(sys.argv))

    write_csv(output_dir / "evidence_keyword_extraction_preview.csv", KEYWORD_PREVIEW_FIELDNAMES, keyword_rows)
    write_csv(output_dir / "chunk_evidence_per_sample.csv", PER_SAMPLE_FIELDNAMES, per_sample_rows)
    write_csv(output_dir / "rerank_topk_evidence_trace.csv", RERANK_TRACE_FIELDNAMES, rerank_trace_rows)
    write_csv(output_dir / "doc_hit_but_evidence_miss.csv", MISS_FIELDNAMES, doc_hit_miss_rows)
    write_json(output_dir / "chunk_evidence_metrics.json", metrics)
    write_json(output_dir / "run_config.json", run_config)
    write_summary(report_dir / "summary.md", metrics, run_config)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset: {path}")
    return [item for item in data if isinstance(item, dict)]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def validate_dataset(samples: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    for sample in samples:
        if bool(sample.get("negative_query")) and sample.get("should_require_doc_hit") is False:
            continue
        if not (sample.get("expected_doc_ids") or sample.get("expected_source_files")):
            errors.append(f"{sample_id(sample)} missing expected_doc_ids/source_files")
    if errors:
        raise ValueError("Dataset validation failed: " + "; ".join(errors[:10]))


def validate_bm25_tokenizer(samples: list[dict[str, Any]]) -> dict[str, Any]:
    empty_count = 0
    single_cjk_count = 0
    preview: list[dict[str, Any]] = []
    for sample in samples[:20]:
        tokens = tokenize_query(str(sample.get("question") or ""))
        if not tokens:
            empty_count += 1
        single_cjk = [token for token in tokens if len(token) == 1 and "\u4e00" <= token <= "\u9fff"]
        single_cjk_count += len(single_cjk)
        if len(preview) < 5:
            preview.append({"sample_id": sample_id(sample), "tokens": tokens[:20]})
    if empty_count:
        raise ValueError(f"BM25 tokenize_query returned empty tokens for {empty_count} preview samples")
    if single_cjk_count > 5:
        raise ValueError(f"BM25 CJK filter appears inactive; single CJK token count={single_cjk_count}")
    return {"empty_query_count": empty_count, "single_cjk_token_count": single_cjk_count, "preview": preview}


def analysis_for_sample(router: QueryRouter, sample: dict[str, Any], question: str) -> QueryAnalysis:
    analysis = router.analyze(question)
    route = str(sample.get("expected_route") or "").strip().lower()
    if route in {"factoid", "summary", "comparison", "experiment", "unknown"}:
        analysis.intent = QueryIntent(route)
    return analysis


def extract_keyword_profile(sample: dict[str, Any]) -> KeywordProfile:
    question = str(sample.get("question") or "")
    answer = reference_answer(sample)
    expected_sections = " ".join(str(item) for item in sample.get("expected_sections") or [])
    question_keywords = extract_keywords(question, include_domain=True)
    answer_keywords = extract_keywords(answer, include_domain=True)
    section_keywords = [normalize_term(item) for item in extract_keywords(expected_sections, include_domain=False)]
    domain_entities = sorted(set(extract_domain_entities(question + " " + answer)))
    answer_keywords = sorted(set(answer_keywords + section_keywords))
    groups = build_normalized_groups(sorted(set(question_keywords + answer_keywords + domain_entities)))
    return KeywordProfile(
        question_keywords=sorted(set(question_keywords)),
        answer_keywords=sorted(set(answer_keywords)),
        domain_entities=domain_entities,
        normalized_groups=groups,
    )


def extract_keywords(text: str, *, include_domain: bool) -> list[str]:
    normalized = normalize_text(text)
    candidates: list[str] = []
    if include_domain:
        candidates.extend(extract_domain_entities(normalized))
    for match in TOKEN_RE.finditer(normalized):
        token = normalize_term(match.group(0))
        if not token or is_stopword(token):
            continue
        if is_cjk(token):
            if token in DOMAIN_ZH_TERMS or 2 <= len(token) <= 6:
                candidates.append(token)
            continue
        if len(token) >= 2 and (looks_domainish(token) or token in DOMAIN_EN_TERMS):
            candidates.append(token)
    return sorted(set(candidates))


def extract_domain_entities(text: str) -> list[str]:
    normalized = normalize_text(text)
    found: list[str] = []
    for term in DOMAIN_ZH_TERMS:
        if term in normalized:
            found.append(term)
    for term in DOMAIN_EN_TERMS:
        if normalize_term(term) in normalized:
            found.append(normalize_term(term))
    return sorted(set(found))


def build_normalized_groups(keywords: list[str]) -> dict[str, list[str]]:
    groups: dict[str, set[str]] = {}
    for keyword in keywords:
        canonical = canonical_term(keyword)
        groups.setdefault(canonical, set()).add(keyword)
        groups[canonical].update(term_aliases(canonical))
    return {key: sorted(values) for key, values in sorted(groups.items())}


def evaluate_stage(
    sample: dict[str, Any],
    profile: KeywordProfile,
    hits: list[RetrievedChunk],
) -> list[tuple[RetrievedChunk, EvidenceEval]]:
    return [(chunk, evaluate_chunk(sample, profile, chunk)) for chunk in hits]


def evaluate_chunk(sample: dict[str, Any], profile: KeywordProfile, chunk: RetrievedChunk) -> EvidenceEval:
    expected_doc = is_expected_doc_chunk(sample, chunk)
    expected_section = is_expected_section_chunk(sample, chunk)
    noise = is_noise_chunk(chunk)
    text_norm = normalize_text(chunk.text)
    matched_answer = matched_keywords(text_norm, profile.answer_keywords, profile.normalized_groups)
    matched_question = matched_keywords(text_norm, profile.question_keywords, profile.normalized_groups)
    text_len = len((chunk.text or "").strip())
    score = 0
    if expected_doc:
        score += 3
    if expected_section:
        score += 2
    score += 2 * len(matched_answer)
    score += len(matched_question)
    if noise:
        score -= 2
    if text_len < 120:
        score -= 1
    answerable = expected_doc and len(matched_answer) >= 1 and text_len >= 120 and not noise
    strong = expected_doc and (len(matched_answer) >= 2 or (expected_section and len(matched_answer) >= 1)) and not noise
    return EvidenceEval(
        expected_doc_chunk=expected_doc,
        expected_section_chunk=expected_section,
        answer_keyword_overlap=len(matched_answer),
        question_keyword_overlap=len(matched_question),
        matched_answer_keywords=matched_answer,
        matched_question_keywords=matched_question,
        evidence_score_simple=score,
        answerable_chunk_weak=answerable,
        strong_evidence_chunk=strong,
        section_noise_flag=noise,
    )


def is_expected_doc_chunk(sample: dict[str, Any], chunk: RetrievedChunk) -> bool:
    expected_docs = set(str(item) for item in sample.get("expected_doc_ids") or [])
    expected_sources = set(str(item) for item in sample.get("expected_source_files") or [])
    return (bool(expected_docs) and chunk.doc_id in expected_docs) or (
        bool(expected_sources) and chunk.source_file in expected_sources
    )


def is_expected_section_chunk(sample: dict[str, Any], chunk: RetrievedChunk) -> bool:
    expected_sections = [normalize_section(item) for item in sample.get("expected_sections") or []]
    if not expected_sections:
        return False
    section_text = normalize_section(
        " ".join(
            str(item)
            for item in [
                chunk.section,
                chunk.metadata.get("section_path", ""),
                chunk.metadata.get("section_group", ""),
            ]
        )
    )
    return any(expected and expected in section_text for expected in expected_sections)


def is_noise_chunk(chunk: RetrievedChunk) -> bool:
    section = " ".join(str(value) for value in [chunk.section, chunk.metadata.get("section_path", "")])
    if NOISE_SECTION_RE.search(section):
        return True
    for key in ("contains_references", "contains_noise", "contains_metadata"):
        if bool(chunk.metadata.get(key)):
            return True
    return False


def matched_keywords(text_norm: str, keywords: list[str], groups: dict[str, list[str]]) -> list[str]:
    matched: list[str] = []
    for keyword in keywords:
        canonical = canonical_term(keyword)
        aliases = groups.get(canonical) or [keyword]
        if any(term_present(text_norm, alias) for alias in aliases):
            matched.append(keyword)
    return sorted(set(matched))


def term_present(text_norm: str, term: str) -> bool:
    term_norm = normalize_term(term)
    if not term_norm:
        return False
    if is_cjk(term_norm):
        return term_norm in text_norm
    return re.search(rf"(?<![a-z0-9]){re.escape(term_norm)}(?![a-z0-9])", text_norm) is not None


def build_keyword_preview_row(sample: dict[str, Any], profile: KeywordProfile) -> dict[str, Any]:
    return {
        "sample_id": sample_id(sample),
        "question": sample.get("question", ""),
        "reference_answer": reference_answer(sample),
        "expected_doc_ids": join_list(sample.get("expected_doc_ids") or []),
        "expected_sections": join_list(sample.get("expected_sections") or []),
        "extracted_question_keywords": join_list(profile.question_keywords),
        "extracted_answer_keywords": join_list(profile.answer_keywords),
        "extracted_domain_entities": join_list(profile.domain_entities),
        "normalized_keyword_groups": json.dumps(profile.normalized_groups, ensure_ascii=False, sort_keys=True),
        "keyword_count": len(set(profile.question_keywords + profile.answer_keywords + profile.domain_entities)),
        "notes": "weak_supervision_keywords",
    }


def build_per_sample_row(
    sample: dict[str, Any],
    skipped_doc_hit: bool,
    stage_evals: dict[str, list[tuple[RetrievedChunk, EvidenceEval]]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": sample_id(sample),
        "question": sample.get("question", ""),
        "expected_doc_ids": join_list(sample.get("expected_doc_ids") or []),
        "expected_source_files": join_list(sample.get("expected_source_files") or []),
        "expected_sections": join_list(sample.get("expected_sections") or []),
        "negative_query": bool(sample.get("negative_query")),
        "skipped_doc_hit": skipped_doc_hit,
        "reference_answer": reference_answer(sample),
    }
    for stage in ("dense", "bm25", "hybrid", "rerank"):
        row[f"{stage}_doc_best_rank"] = best_rank(stage_evals[stage], lambda ev: ev.expected_doc_chunk)
        row[f"{stage}_best_answerable_chunk_rank"] = best_rank(stage_evals[stage], lambda ev: ev.answerable_chunk_weak)
        row[f"{stage}_best_strong_evidence_rank"] = best_rank(stage_evals[stage], lambda ev: ev.strong_evidence_chunk)
    for stage in ("dense", "bm25", "hybrid"):
        for k in (10, 20, 40):
            row[f"{stage}_answerable_hit_at_{k}"] = hit_at(stage_evals[stage], lambda ev: ev.answerable_chunk_weak, k, skipped_doc_hit)
    for k in (5, 10, 20):
        row[f"rerank_answerable_hit_at_{k}"] = hit_at(
            stage_evals["rerank"], lambda ev: ev.answerable_chunk_weak, k, skipped_doc_hit
        )
        row[f"rerank_strong_evidence_hit_at_{k}"] = hit_at(
            stage_evals["rerank"], lambda ev: ev.strong_evidence_chunk, k, skipped_doc_hit
        )
    row["doc_hit_but_evidence_miss"] = (
        bool(row["rerank_doc_best_rank"] and int(row["rerank_doc_best_rank"]) <= 10)
        and not as_bool(row["rerank_answerable_hit_at_10"])
        and not skipped_doc_hit
    )
    row["evidence_found_but_reranker_suppressed"] = (
        as_bool(row["hybrid_answerable_hit_at_20"]) and not as_bool(row["rerank_answerable_hit_at_20"]) and not skipped_doc_hit
    )
    row["evidence_found_in_hybrid_but_not_rerank"] = row["evidence_found_but_reranker_suppressed"]
    status, action = diagnose_evidence(row, skipped_doc_hit)
    row["evidence_retrieval_status"] = status
    row["recommended_next_action"] = action
    return row


def build_rerank_trace_rows(
    sample: dict[str, Any],
    profile: KeywordProfile,
    reranked_hits: list[RetrievedChunk],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, chunk in enumerate(reranked_hits[:20], start=1):
        ev = evaluate_chunk(sample, profile, chunk)
        rows.append(
            {
                "sample_id": sample_id(sample),
                "rerank_rank": rank,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source_file": chunk.source_file,
                "section": chunk.section,
                "section_path": chunk.metadata.get("section_path", ""),
                "rerank_score": format_score(chunk.rerank_score),
                "dense_score_if_available": format_score(chunk.vector_score),
                "bm25_score_if_available": format_score(chunk.bm25_score),
                "hybrid_score_if_available": format_score(chunk.fusion_score),
                "expected_doc_chunk": ev.expected_doc_chunk,
                "expected_section_chunk": ev.expected_section_chunk,
                "answer_keyword_overlap": ev.answer_keyword_overlap,
                "question_keyword_overlap": ev.question_keyword_overlap,
                "matched_answer_keywords": join_list(ev.matched_answer_keywords),
                "matched_question_keywords": join_list(ev.matched_question_keywords),
                "evidence_score_simple": ev.evidence_score_simple,
                "answerable_chunk_weak": ev.answerable_chunk_weak,
                "strong_evidence_chunk": ev.strong_evidence_chunk,
                "section_noise_flag": ev.section_noise_flag,
                "text_preview": compact_text(chunk.text, 260),
            }
        )
    return rows


def build_doc_hit_but_evidence_miss_row(
    sample: dict[str, Any],
    profile: KeywordProfile,
    reranked_hits: list[RetrievedChunk],
    stage_evals: dict[str, list[tuple[RetrievedChunk, EvidenceEval]]],
    per_row: dict[str, Any],
) -> dict[str, Any] | None:
    if not as_bool(per_row["doc_hit_but_evidence_miss"]):
        return None
    expected_doc_chunks = [(chunk, ev) for chunk, ev in stage_evals["rerank"][:10] if ev.expected_doc_chunk]
    sections = sorted(set(chunk.section for chunk, _ev in expected_doc_chunks))
    why = "unclear"
    if not expected_doc_chunks:
        why = "answer_keywords_absent_from_retrieved_chunks"
    elif any(ev.section_noise_flag for _chunk, ev in expected_doc_chunks):
        why = "expected_doc_chunk_in_references_or_noise"
    elif any(len(chunk.text or "") < 120 for chunk, _ev in expected_doc_chunks):
        why = "expected_doc_chunk_too_short"
    elif not any(ev.expected_section_chunk for _chunk, ev in expected_doc_chunks) and sample.get("expected_sections"):
        why = "expected_doc_chunk_in_wrong_section"
    elif not any(ev.answer_keyword_overlap for _chunk, ev in expected_doc_chunks):
        why = "answer_keywords_absent_from_retrieved_chunks"
    elif len(profile.answer_keywords) < 2:
        why = "keyword_extraction_weak"
    else:
        why = "expected_doc_chunk_not_answer_bearing"
    return {
        "sample_id": sample_id(sample),
        "question": sample.get("question", ""),
        "expected_doc_ids": join_list(sample.get("expected_doc_ids") or []),
        "rerank_doc_best_rank": per_row["rerank_doc_best_rank"],
        "expected_doc_chunks_in_rerank_top10": join_list([chunk.chunk_id for chunk, _ev in expected_doc_chunks]),
        "expected_doc_chunk_sections": join_list(sections),
        "extracted_answer_keywords": join_list(profile.answer_keywords),
        "extracted_question_keywords": join_list(profile.question_keywords),
        "why_evidence_miss": why,
        "recommended_next_action": "doc_local_evidence_selection" if expected_doc_chunks else "chunk_level_retrieval_improvement",
    }


def best_rank(stage_rows: list[tuple[RetrievedChunk, EvidenceEval]], predicate: Any) -> int:
    for rank, (_chunk, ev) in enumerate(stage_rows, start=1):
        if predicate(ev):
            return rank
    return 0


def hit_at(stage_rows: list[tuple[RetrievedChunk, EvidenceEval]], predicate: Any, k: int, skipped: bool) -> bool:
    if skipped:
        return False
    return best_rank(stage_rows[:k], predicate) > 0


def diagnose_evidence(row: dict[str, Any], skipped: bool) -> tuple[str, str]:
    if skipped:
        return "skipped_negative", "skipped_negative"
    if as_bool(row["rerank_strong_evidence_hit_at_10"]):
        return "strong_evidence_in_rerank_top10", "generation_lifecycle_debug"
    if as_bool(row["rerank_answerable_hit_at_10"]):
        return "weak_evidence_in_rerank_top10", "generation_lifecycle_debug"
    if as_bool(row["hybrid_answerable_hit_at_20"]) and not as_bool(row["rerank_answerable_hit_at_20"]):
        return "evidence_found_before_rerank_but_dropped", "reranker_evidence_ranking_audit"
    if row["rerank_doc_best_rank"] and int(row["rerank_doc_best_rank"]) <= 10:
        return "doc_hit_but_no_answerable_chunk", "doc_local_evidence_selection"
    if not any(int(row[f"{stage}_best_answerable_chunk_rank"]) > 0 for stage in ("dense", "bm25", "hybrid", "rerank")):
        return "hard_evidence_miss", "chunk_level_retrieval_improvement"
    return "unclear", "manual_review"


def compute_metrics(
    samples: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    phase16r_metrics: dict[str, Any],
) -> dict[str, Any]:
    evaluated = [row for row in rows if not as_bool(row["skipped_doc_hit"])]
    skipped_count = len(rows) - len(evaluated)
    metrics: dict[str, Any] = {
        "total_samples": len(rows),
        "evaluated_samples": len(evaluated),
        "skipped_negative_query_count": skipped_count,
        "rerank_doc_hit_at_5": phase16r_metrics.get("rerank_doc_hit_at_5"),
        "rerank_doc_hit_at_10": phase16r_metrics.get("rerank_doc_hit_at_10"),
        "rerank_doc_hit_at_20": phase16r_metrics.get("rerank_doc_hit_at_20"),
    }
    for stage in ("dense", "bm25", "hybrid"):
        for k in (10, 20, 40):
            metrics[f"{stage}_answerable_chunk_hit_at_{k}"] = rate(evaluated, f"{stage}_answerable_hit_at_{k}")
    for k in (5, 10, 20):
        metrics[f"rerank_answerable_chunk_hit_at_{k}"] = rate(evaluated, f"rerank_answerable_hit_at_{k}")
        metrics[f"rerank_strong_evidence_hit_at_{k}"] = rate(evaluated, f"rerank_strong_evidence_hit_at_{k}")
    metrics["rerank_doc_hit_but_answerable_chunk_miss_count"] = sum(as_bool(row["doc_hit_but_evidence_miss"]) for row in evaluated)
    metrics["rerank_doc_hit_but_strong_evidence_miss_count"] = sum(
        int(row["rerank_doc_best_rank"]) > 0
        and int(row["rerank_doc_best_rank"]) <= 10
        and not as_bool(row["rerank_strong_evidence_hit_at_10"])
        for row in evaluated
    )
    metrics["evidence_found_in_hybrid_but_dropped_by_rerank_count"] = sum(
        as_bool(row["evidence_found_in_hybrid_but_not_rerank"]) for row in evaluated
    )
    metrics["hard_evidence_miss_count"] = sum(row["evidence_retrieval_status"] == "hard_evidence_miss" for row in evaluated)
    status_distribution = Counter(row["evidence_retrieval_status"] for row in rows)
    metrics["evidence_retrieval_status_distribution"] = dict(status_distribution)
    add_comparison_metrics(metrics, samples, rows)
    doc10 = float(metrics.get("rerank_doc_hit_at_10") or 0.0)
    evidence10 = float(metrics["rerank_answerable_chunk_hit_at_10"])
    metrics["doc_level_vs_evidence_level_gap"] = round(doc10 - evidence10, 4)
    metrics["retrieval_is_sufficient_for_generation"] = evidence10 >= 0.75 and (doc10 - evidence10) <= 0.15
    metrics["recommended_next_phase"] = recommend_next_phase(metrics)
    return metrics


def add_comparison_metrics(metrics: dict[str, Any], samples: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    pairs = [
        (sample, row)
        for sample, row in zip(samples, rows)
        if not as_bool(row["skipped_doc_hit"]) and str(sample.get("expected_route")) == "comparison" and sample.get("expected_doc_ids")
    ]
    denom = max(len(pairs), 1)
    any_answerable = 0
    all_answerable = 0
    any_strong = 0
    all_strong = 0
    for sample, _row in pairs:
        expected_docs = [str(item) for item in sample.get("expected_doc_ids") or []]
        sample_id_value = sample_id(sample)
        trace_rows = read_current_rerank_trace_cache(sample_id_value)
        answerable_docs = {
            row["doc_id"] for row in trace_rows if as_bool(row["answerable_chunk_weak"]) and int(row["rerank_rank"]) <= 10
        }
        strong_docs = {
            row["doc_id"] for row in trace_rows if as_bool(row["strong_evidence_chunk"]) and int(row["rerank_rank"]) <= 10
        }
        expected_set = set(expected_docs)
        any_answerable += bool(expected_set & answerable_docs)
        all_answerable += bool(expected_set) and expected_set.issubset(answerable_docs)
        any_strong += bool(expected_set & strong_docs)
        all_strong += bool(expected_set) and expected_set.issubset(strong_docs)
    metrics["comparison_answerable_any_hit_at_10"] = round(any_answerable / denom, 4)
    metrics["comparison_answerable_all_branch_hit_at_10"] = round(all_answerable / denom, 4)
    metrics["comparison_strong_evidence_any_hit_at_10"] = round(any_strong / denom, 4)
    metrics["comparison_strong_evidence_all_branch_hit_at_10"] = round(all_strong / denom, 4)


_RERANK_TRACE_CACHE: dict[str, list[dict[str, Any]]] = {}


def read_current_rerank_trace_cache(sample_id_value: str) -> list[dict[str, Any]]:
    return _RERANK_TRACE_CACHE.get(sample_id_value, [])


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return round(sum(as_bool(row[field]) for row in rows) / max(len(rows), 1), 4)


def recommend_next_phase(metrics: dict[str, Any]) -> str:
    evidence10 = float(metrics["rerank_answerable_chunk_hit_at_10"])
    gap = float(metrics["doc_level_vs_evidence_level_gap"])
    dropped = int(metrics["evidence_found_in_hybrid_but_dropped_by_rerank_count"])
    hard = int(metrics["hard_evidence_miss_count"])
    if evidence10 >= 0.75 and gap <= 0.15:
        return "Phase 16B evidence lifecycle debug/drop_reason"
    if dropped > hard:
        return "reranker evidence ranking audit"
    return "chunk-level retrieval / doc-local evidence selection"


def build_run_config(args: argparse.Namespace, settings: Settings, dataset_path: Path, command_used: str) -> dict[str, Any]:
    return {
        "branch": git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit_sha": git_output(["git", "rev-parse", "HEAD"]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "chunks_path": settings.kb.chunk_jsonl,
        "bm25_query_tokenizer": "tokenize_query_cjk_filtered",
        "dense_model": settings.kb.embedding_model_path,
        "reranker_model": settings.reranker.model_path or settings.reranker.model_name,
        "dense_top_k": args.dense_top_k,
        "bm25_top_k": args.bm25_top_k,
        "hybrid_top_k": args.hybrid_top_k,
        "rerank_top_k": args.rerank_top_k,
        "generation_called": False,
        "qwen_called": False,
        "qwen_synthesis": False,
        "support_selector_called": False,
        "citation_binder_called": False,
        "parent_expansion_used_for_main_metric": False,
        "targeted_filter_used": False,
        "evidence_matching_mode": "weak_supervision",
        "generation_version": "v2",
        "generation_v2_enable_comparison_coverage": False,
        "generation_v2_enable_neighbor_audit": False,
        "generation_v2_enable_neighbor_promotion": False,
        "generation_v2_include_neighbor_context_in_qwen": False,
        "retrieval_biolexical_bm25_enabled": os.environ.get("RETRIEVAL_BIOLEXICAL_BM25_ENABLED", "false"),
        "command_used": command_used,
    }


def write_summary(path: Path, metrics: dict[str, Any], run_config: dict[str, Any]) -> None:
    status = metrics["evidence_retrieval_status_distribution"]
    lines = [
        "# Phase 16R-2 Chunk / Evidence-level Retrieval Audit",
        "",
        "## 1. Purpose",
        "",
        "Phase 16R measured doc_hit only. This phase keeps the same open retrieval-only boundary and checks whether dense/BM25/hybrid/rerank surface answer-bearing chunks, not just the expected document.",
        "",
        "## 2. Method",
        "",
        "This is a weak-supervision evidence audit, not a human gold-chunk benchmark. A chunk is scored using expected_doc/source_file match, expected_section match, reference-answer keywords, question/domain terms, and simple noise penalties.",
        "",
        "- answerable_chunk_weak: expected doc/source chunk, at least one answer keyword overlap, sufficient text length, and not references/noise.",
        "- strong_evidence_chunk: expected doc/source chunk with at least two answer keyword overlaps, or expected section plus at least one answer keyword overlap.",
        "- evidence_score_simple: +3 expected doc, +2 expected section, +2 per answer keyword, +1 per question keyword, -2 references/noise, -1 short text.",
        "",
        "## 3. Main Metrics",
        "",
        "| Metric | @5 | @10 | @20 | @40 |",
        "|---|---:|---:|---:|---:|",
        f"| rerank doc_hit | {metrics.get('rerank_doc_hit_at_5')} | {metrics.get('rerank_doc_hit_at_10')} | {metrics.get('rerank_doc_hit_at_20')} | - |",
        f"| rerank answerable_chunk_hit | {metrics['rerank_answerable_chunk_hit_at_5']} | {metrics['rerank_answerable_chunk_hit_at_10']} | {metrics['rerank_answerable_chunk_hit_at_20']} | - |",
        f"| rerank strong_evidence_hit | {metrics['rerank_strong_evidence_hit_at_5']} | {metrics['rerank_strong_evidence_hit_at_10']} | {metrics['rerank_strong_evidence_hit_at_20']} | - |",
        f"| dense answerable_chunk_hit | - | {metrics['dense_answerable_chunk_hit_at_10']} | {metrics['dense_answerable_chunk_hit_at_20']} | {metrics['dense_answerable_chunk_hit_at_40']} |",
        f"| BM25 answerable_chunk_hit | - | {metrics['bm25_answerable_chunk_hit_at_10']} | {metrics['bm25_answerable_chunk_hit_at_20']} | {metrics['bm25_answerable_chunk_hit_at_40']} |",
        f"| hybrid answerable_chunk_hit | - | {metrics['hybrid_answerable_chunk_hit_at_10']} | {metrics['hybrid_answerable_chunk_hit_at_20']} | {metrics['hybrid_answerable_chunk_hit_at_40']} |",
        "",
        "## 4. Doc-level vs Evidence-level Gap",
        "",
        f"- rerank_doc_hit_but_answerable_chunk_miss_count: {metrics['rerank_doc_hit_but_answerable_chunk_miss_count']}",
        f"- evidence_found_in_hybrid_but_dropped_by_rerank_count: {metrics['evidence_found_in_hybrid_but_dropped_by_rerank_count']}",
        f"- hard_evidence_miss_count: {metrics['hard_evidence_miss_count']}",
        f"- doc_level_vs_evidence_level_gap@10: {metrics['doc_level_vs_evidence_level_gap']}",
        "",
        "## 5. Failure Groups",
        "",
        f"- strong_evidence_in_rerank_top10: {status.get('strong_evidence_in_rerank_top10', 0)}",
        f"- weak_evidence_in_rerank_top10: {status.get('weak_evidence_in_rerank_top10', 0)}",
        f"- doc_hit_but_no_answerable_chunk: {status.get('doc_hit_but_no_answerable_chunk', 0)}",
        f"- evidence_found_before_rerank_but_dropped: {status.get('evidence_found_before_rerank_but_dropped', 0)}",
        f"- hard_evidence_miss: {status.get('hard_evidence_miss', 0)}",
        f"- skipped_negative: {status.get('skipped_negative', 0)}",
        "",
        "## 6. Interpretation",
        "",
        interpretation(metrics),
        "",
        "## 7. Recommendation",
        "",
        f"Recommended next phase: {metrics['recommended_next_phase']}.",
        "",
        f"Run config: open retrieval=true; generation_called={run_config['generation_called']}; qwen_called={run_config['qwen_called']}; support_selector_called={run_config['support_selector_called']}; citation_binder_called={run_config['citation_binder_called']}; parent_expansion_used_for_main_metric={run_config['parent_expansion_used_for_main_metric']}.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def interpretation(metrics: dict[str, Any]) -> str:
    doc10 = float(metrics.get("rerank_doc_hit_at_10") or 0.0)
    evidence10 = float(metrics["rerank_answerable_chunk_hit_at_10"])
    strong10 = float(metrics["rerank_strong_evidence_hit_at_10"])
    gap = float(metrics["doc_level_vs_evidence_level_gap"])
    dropped = int(metrics["evidence_found_in_hybrid_but_dropped_by_rerank_count"])
    hard = int(metrics["hard_evidence_miss_count"])
    if evidence10 >= 0.75 and gap <= 0.15:
        main = "Doc-level retrieval does not materially overestimate evidence retrieval under the weak-supervision rule."
    else:
        main = "Doc-level retrieval overestimates evidence retrieval; chunk-level evidence selection remains a material issue."
    return (
        f"rerank_doc_hit@10={doc10:.4f}, rerank_answerable_chunk_hit@10={evidence10:.4f}, "
        f"rerank_strong_evidence_hit@10={strong10:.4f}, gap={gap:.4f}. {main} "
        f"Reranker-dropped evidence count is {dropped}; hard evidence miss count is {hard}. "
        "Parent expansion was not used for main metrics; a later parent-expansion evidence audit may be useful only if this pre-expansion audit shows a large doc/evidence gap."
    )


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.translate(str.maketrans({"\u2032": "'", "\u2018": "'", "\u2019": "'", "\u02bc": "'", "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-"}))
    value = value.replace("α", "alpha").replace("β", "beta").replace("γ", "gamma")
    value = value.replace("Α", "alpha").replace("Β", "beta").replace("Γ", "gamma")
    return re.sub(r"\s+", " ", value.lower()).strip()


def normalize_term(term: Any) -> str:
    value = normalize_text(str(term or ""))
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ;:,.()[]{}\"")


def normalize_section(section: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", normalize_text(str(section or ""))).strip()


def canonical_term(term: str) -> str:
    value = normalize_term(term)
    aliases = {
        "2-fl": "2'-fl",
        "2 fucosyllactose": "2'-fl",
        "2-fucosyllactose": "2'-fl",
        "6-sl": "6'-sl",
        "6 sialyllactose": "6'-sl",
        "6-sialyllactose": "6'-sl",
        "6'-sialyllactose": "6'-sl",
        "alpha": "alpha",
        "beta": "beta",
    }
    return aliases.get(value, value)


def term_aliases(canonical: str) -> set[str]:
    aliases = {canonical}
    if canonical == "2'-fl":
        aliases.update({"2-fl", "2'-fucosyllactose", "2-fucosyllactose"})
    if canonical == "6'-sl":
        aliases.update({"6-sl", "6'-sialyllactose", "6-sialyllactose"})
    return aliases


def looks_domainish(token: str) -> bool:
    if any(ch.isdigit() for ch in token):
        return True
    if any(ch in token for ch in ["'", "-", "_", "."]):
        return True
    if token.isupper() and len(token) >= 2:
        return True
    if token in DOMAIN_EN_TERMS:
        return True
    return len(token) >= 4 and token not in STOPWORDS_EN


def is_stopword(token: str) -> bool:
    return token in STOPWORDS_ZH or token in STOPWORDS_EN


def is_cjk(token: str) -> bool:
    return bool(token) and all("\u4e00" <= ch <= "\u9fff" for ch in token)


def reference_answer(sample: dict[str, Any]) -> str:
    for key in ("reference_answer", "ground_truth", "expected_answer", "answer", "reference"):
        value = sample.get(key)
        if value:
            return str(value)
    return ""


def sample_id(sample: dict[str, Any]) -> str:
    return str(sample.get("id") or sample.get("sample_id") or "")


def join_list(values: Any) -> str:
    if isinstance(values, dict):
        return json.dumps(values, ensure_ascii=False, sort_keys=True)
    return "|".join(str(value) for value in values)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def compact_text(text: str, limit: int) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit]


def format_score(value: Any) -> str:
    try:
        return f"{float(value or 0.0):.6f}"
    except Exception:
        return "0.000000"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_self_tests() -> None:
    sample = {
        "id": "test",
        "question": "2′-FL 与 6′-SL 在 Pichia 中的产量如何？",
        "reference": "2′-FL、6′-sialyllactose 和 Pichia 是关键实体，中文泛词总结不应保留。",
        "expected_doc_ids": ["doc_x"],
        "expected_sections": ["Results"],
    }
    profile = extract_keyword_profile(sample)
    assert "2'-fl" in profile.answer_keywords or "2'-fl" in profile.domain_entities
    assert "6'-sl" in profile.normalized_groups
    assert "总结" not in profile.question_keywords
    good = RetrievedChunk(
        chunk_id="c1", doc_id="doc_x", source_file="doc_x.pdf", title="", section="Results",
        text=(
            "Results show 2'-FL and 6'-sialyllactose production in Pichia increased significantly. "
            "The engineered strain improved secretion and product titer under the tested fermentation condition."
        ),
    )
    wrong_doc = RetrievedChunk(
        chunk_id="c2", doc_id="doc_y", source_file="doc_y.pdf", title="", section="Results",
        text="2'-FL and Pichia are mentioned here.",
    )
    refs = RetrievedChunk(
        chunk_id="c3", doc_id="doc_x", source_file="doc_x.pdf", title="", section="References",
        text="References about 2'-FL and 6'-sialyllactose in Pichia.",
    )
    assert evaluate_chunk(sample, profile, good).answerable_chunk_weak
    assert not evaluate_chunk(sample, profile, wrong_doc).answerable_chunk_weak
    refs_eval = evaluate_chunk(sample, profile, refs)
    assert refs_eval.section_noise_flag and not refs_eval.answerable_chunk_weak
    print("self-tests passed")


if __name__ == "__main__":
    main()
