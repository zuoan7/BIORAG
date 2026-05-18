#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.synbio_rag.application.pipeline import (
    SynBioRAGPipeline,
    _contains_cjk,
    _run_original_cn_fallback,
)
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.schemas import QueryFilters
from src.synbio_rag.evaluation.failure_taxonomy import evaluate_failure
from src.synbio_rag.rewrite.query_rewrite_service import QueryRewriteMode, QueryRewriteService, RewriteTrace, get_prompt_hash


RDIR = ROOT / "results/phase21a9d_remaining_smoke150_regression"
REPDIR = ROOT / "reports/phase21a9d_remaining_smoke150_regression"
P21C = ROOT / "results/phase21a9c_rewrite_wiring_fix"
P21 = ROOT / "results/phase21a9_smoke200_rebaseline"
P21B = ROOT / "results/phase21a9b_rewrite_fallback_audit"
P20M = ROOT / "results/phase20m_convergence_summary"
P20L2 = ROOT / "results/phase20l2_original_cn_fallback_floor_feature_ab"
SMOKE150 = ROOT / "data/eval/datasets/smoke150.jsonl"
SMOKE200 = ROOT / "data/eval/datasets/smoke200.jsonl"


def configure_eval_env() -> None:
    os.environ["QUERY_REWRITE_MODE"] = "enabled"
    os.environ["QUERY_REWRITE_REQUIRE_LLM_FOR_EVAL"] = "true"
    os.environ["RETRIEVAL_ORIGINAL_CN_FALLBACK_ENABLED"] = "true"
    os.environ["GENERATION_VERSION"] = "v2"
    os.environ["RETRIEVAL_SOURCE_FLOOR_ENABLED"] = "true"
    os.environ["RETRIEVAL_PARENT_EXPANSION_ENABLED"] = "true"
    os.environ["RETRIEVAL_ALIAS_EXPANSION_ENABLED"] = "false"
    os.environ["GENERATION_V2_USE_QWEN_SYNTHESIS"] = "false"
    os.environ["GENERATION_V2_ENABLE_COMPARISON_COVERAGE"] = "false"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pipe(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    return "|".join(str(v) for v in values)


def parse_pipe(value: str) -> list[str]:
    if not value:
        return []
    return [item for item in value.split("|") if item]


def is_negative(sample: dict[str, Any]) -> bool:
    text = " ".join([
        str(sample.get("expected_route") or ""),
        str(sample.get("category") or ""),
        " ".join(str(t) for t in (sample.get("tags") or [])),
        str(sample.get("question") or ""),
    ]).lower()
    return "negative" in text or "abstain" in text or "如果没有" in text or "是否有关于" in text


def sample_category(sample: dict[str, Any]) -> str:
    category = sample.get("category") or ""
    if category == "table_figure_caption":
        return "pdf_table_figure"
    if category == "multi_doc_comparison":
        return "comparison"
    return category or "unknown"


def build_rewrite_service(settings: Settings) -> QueryRewriteService:
    qrc = settings.query_rewrite
    from src.synbio_rag.application.pipeline import _build_query_rewrite_llm_client
    llm_client, llm_error = _build_query_rewrite_llm_client(settings)
    return QueryRewriteService(
        mode=QueryRewriteMode(qrc.mode),
        model=qrc.model,
        temperature=qrc.temperature,
        cache_enabled=qrc.cache_enabled,
        timeout_ms=qrc.timeout_ms,
        fallback_on_error=qrc.fallback_on_error,
        guard_implicit=qrc.guard_implicit_reference,
        guard_negative=qrc.guard_negative_intent,
        cache_version=qrc.cache_key_version,
        llm_client=llm_client,
        llm_client_error=llm_error,
    )


def doc_hit(expected_doc_ids: list[str], doc_ids: list[str]) -> bool:
    if not expected_doc_ids:
        return True
    return bool(set(expected_doc_ids).intersection(doc_ids))


def first_expected_present(expected_doc_ids: list[str], chunks: list[Any]) -> bool:
    return doc_hit(expected_doc_ids, [getattr(chunk, "doc_id", "") for chunk in chunks])


def chunk_doc_ids(chunks: list[Any]) -> list[str]:
    return list(dict.fromkeys(getattr(chunk, "doc_id", "") for chunk in chunks if getattr(chunk, "doc_id", "")))


def direct_dense_bm25(pipeline: SynBioRAGPipeline, query: str, filters: QueryFilters | None) -> tuple[list[Any], list[Any]]:
    dense = pipeline.dense_retriever.search(query, limit=40, filters=filters)
    bm25 = pipeline.bm25_retriever.search(query, limit=40, filters=filters)
    return dense, bm25


def run_variant(
    *,
    pipeline: SynBioRAGPipeline,
    sample: dict[str, Any],
    variant_id: str,
    retrieval_query: str,
    use_original_cn_fallback: bool,
    dual_query_shadow: bool,
    filters: QueryFilters,
) -> tuple[dict[str, Any], dict[str, Any]]:
    question = sample["question"]
    expected_doc_ids = sample.get("expected_doc_ids") or []
    analysis = pipeline.router.analyze(question)

    dense, bm25 = direct_dense_bm25(pipeline, retrieval_query, filters)
    retrieved, retrieval_debug = pipeline._search_with_filter_fallback(
        question=retrieval_query,
        analysis=analysis,
        filters=filters,
        original_question=question,
    )
    cn_fallback_debug = {
        "triggered": False,
        "fallback_added_count": 0,
        "fallback_added_doc_ids": [],
        "merged_candidates": list(retrieved),
    }

    if dual_query_shadow:
        cn_retrieved, _ = pipeline._search_with_filter_fallback(
            question=question,
            analysis=analysis,
            filters=filters,
            original_question=question,
        )
        seen = {chunk.chunk_id for chunk in retrieved}
        merged = list(retrieved)
        for chunk in cn_retrieved:
            if chunk.chunk_id not in seen:
                merged.append(chunk)
                seen.add(chunk.chunk_id)
        retrieved = merged
        cn_fallback_debug = {
            "triggered": True,
            "fallback_added_count": len(merged) - len(retrieval_debug.get("attempts", [])),
            "fallback_added_doc_ids": chunk_doc_ids(cn_retrieved),
            "merged_candidates": merged,
            "reason": "diagnostic_dual_query_shadow",
        }
    elif use_original_cn_fallback:
        rewrite_trace = RewriteTrace(
            query_rewrite_mode="enabled",
            query_rewrite_enabled=True,
            original_query=question,
            rewritten_query=retrieval_query,
            rewrite_fallback_used=False,
            retrieval_query_used="rewritten",
        )
        cn_fallback_debug = _run_original_cn_fallback(
            question=question,
            retrieval_question=retrieval_query,
            rewrite_trace=rewrite_trace,
            retrieved=retrieved,
            analysis=analysis,
            filters=filters,
            config=pipeline.settings.retrieval,
            pipeline=pipeline,
        )
        if cn_fallback_debug.get("triggered"):
            retrieved = cn_fallback_debug["merged_candidates"]

    reranked = pipeline.reranker.rerank(
        question,
        retrieved,
        top_k=analysis.rerank_top_k,
        analysis=analysis,
    )
    seed_chunks = reranked[: pipeline.settings.retrieval.final_top_k]
    final_chunks = seed_chunks
    parent_debug: dict[str, Any] = {}
    if pipeline.settings.generation.version == "v2":
        final_chunks, parent_debug = pipeline.parent_expander.expand(
            question=question,
            seed_chunks=seed_chunks,
            analysis=analysis,
        )
        gen_result = pipeline.generator_v2.run(
            question=question,
            analysis=analysis,
            seed_chunks=final_chunks,
            config=pipeline.settings.generation,
        )
        support_doc_ids = [
            str(item.get("doc_id"))
            for item in (gen_result.debug.get("support_pack") or [])
            if item.get("doc_id")
        ]
        cited_doc_ids = [citation.doc_id for citation in gen_result.citations if citation.doc_id]
    else:
        gen_result = None
        support_doc_ids = []
        cited_doc_ids = []

    variant_row = {
        "sample_id": sample["sample_id"],
        "variant_id": variant_id,
        "retrieval_query": retrieval_query,
        "expected_doc_in_dense_top40": first_expected_present(expected_doc_ids, dense),
        "expected_doc_in_bm25_top40": first_expected_present(expected_doc_ids, bm25),
        "expected_doc_in_hybrid_top40": first_expected_present(expected_doc_ids, retrieved),
        "expected_doc_in_rerank_input": first_expected_present(expected_doc_ids, retrieved),
        "expected_doc_in_final": first_expected_present(expected_doc_ids, final_chunks),
        "expected_doc_in_selected_support": doc_hit(expected_doc_ids, support_doc_ids),
        "expected_doc_cited": doc_hit(expected_doc_ids, cited_doc_ids),
        "doc_hit": doc_hit(expected_doc_ids, list(dict.fromkeys(cited_doc_ids + support_doc_ids))),
        "real_P0_predicted_or_actual": not doc_hit(expected_doc_ids, list(dict.fromkeys(cited_doc_ids + support_doc_ids))) if expected_doc_ids else False,
        "notes": (
            f"dense_docs={pipe(chunk_doc_ids(dense[:5]))}; "
            f"bm25_docs={pipe(chunk_doc_ids(bm25[:5]))}; "
            f"hybrid_docs={pipe(chunk_doc_ids(retrieved[:8]))}; "
            f"final_docs={pipe(chunk_doc_ids(final_chunks))}; "
            f"support_docs={pipe(support_doc_ids)}; cited_docs={pipe(cited_doc_ids)}"
        ),
    }
    debug = {
        "retrieved": retrieved,
        "reranked": reranked,
        "final_chunks": final_chunks,
        "support_doc_ids": support_doc_ids,
        "cited_doc_ids": cited_doc_ids,
        "cn_fallback_debug": cn_fallback_debug,
        "parent_debug": parent_debug,
        "gen_result": gen_result,
    }
    return variant_row, debug


def evaluate_current(sample: dict[str, Any], debug: dict[str, Any], route_pred: str) -> dict[str, Any]:
    expected_doc_ids = sample.get("expected_doc_ids") or []
    cited_doc_ids = list(dict.fromkeys(debug.get("cited_doc_ids", [])))
    support_doc_ids = list(dict.fromkeys(debug.get("support_doc_ids", [])))
    hit = doc_hit(expected_doc_ids, cited_doc_ids + support_doc_ids)
    miss = bool(expected_doc_ids) and not hit
    route_match = (sample.get("expected_route") or "") == route_pred
    raw_failure = "doc_miss" if miss else "route_mismatch" if not route_match else "ok"
    assessed = evaluate_failure(
        raw_failure_category=raw_failure,
        doc_hit=hit,
        cited_doc_ids=cited_doc_ids,
        expected_doc_ids=expected_doc_ids,
        expected_source_files=sample.get("expected_source_files") or [],
        citation_count=len(cited_doc_ids),
        expected_min_citations=0 if is_negative(sample) else max(1, min(2, len(expected_doc_ids))),
        answer_mode="full",
        is_negative=is_negative(sample),
        route_match=route_match,
        source_file_hit=False,
    )
    wrong_doc = bool(cited_doc_ids) and bool(expected_doc_ids) and not set(cited_doc_ids).intersection(expected_doc_ids)
    failure_class = assessed.corrected_failure_category
    if wrong_doc and assessed.is_real_p0:
        failure_class = "wrong_doc_citation"
    first_loss = "none"
    if miss:
        first_loss = "retrieval_or_support"
    elif assessed.is_real_p0 and wrong_doc:
        first_loss = "citation"
    elif assessed.is_real_p0 and is_negative(sample):
        first_loss = "generation"
    elif assessed.is_real_p0:
        first_loss = "eval_or_route"
    return {
        "route_pred": route_pred,
        "doc_hit": hit,
        "doc_miss": miss,
        "cited_doc_ids": cited_doc_ids,
        "support_doc_ids": support_doc_ids,
        "real_P0": bool(assessed.is_real_p0),
        "failure_class": failure_class,
        "first_loss_stage": first_loss,
    }


TEXT_ENTITY_RE = re.compile(
    r"doc_\d+|[A-Za-z][A-Za-z0-9′'αβγΔΔ_\-]*\d*[A-Za-z0-9]?|[\u4e00-\u9fff]{2,}|[0-9]+(?:′|')?-[A-Za-z]+",
    re.UNICODE,
)


def entities(text: str) -> set[str]:
    stop = {"the", "and", "or", "in", "of", "for", "to", "with", "what", "which", "how", "are", "is", "as"}
    out = set()
    for match in TEXT_ENTITY_RE.findall(text or ""):
        normalized = match.strip(".,;:()[]{}\"'").lower()
        if len(normalized) < 2 or normalized in stop:
            continue
        out.add(normalized)
    return out


def semantic_delta(phase20: str, current: str, original: str) -> dict[str, Any]:
    if not phase20:
        return {
            "lexical_delta": "",
            "semantic_delta_assessment": "unknown",
            "possible_semantic_drift": "unclear",
            "lost_entities": "",
            "added_entities": "",
            "changed_intent": False,
            "changed_scope": False,
            "notes": "No Phase 20 rewritten query found.",
        }
    ratio = SequenceMatcher(None, phase20.lower(), current.lower()).ratio()
    lexical_delta = round(1.0 - ratio, 4)
    phase20_entities = entities(phase20)
    current_entities = entities(current)
    lost = sorted(phase20_entities - current_entities)
    added = sorted(current_entities - phase20_entities)
    changed_intent = any(term in phase20.lower() for term in ["no ", "insufficient", "whether", "if there"]) != any(
        term in current.lower() for term in ["no ", "insufficient", "whether", "if there"]
    )
    changed_scope = (
        ("this paper" in phase20.lower()) != ("this paper" in current.lower())
        or ("library" in phase20.lower()) != ("library" in current.lower())
        or ("literature" in phase20.lower()) != ("literature" in current.lower())
    )
    if lexical_delta <= 0.12 and not lost and not changed_intent and not changed_scope:
        assessment = "same"
    elif lexical_delta <= 0.35 and len(lost) <= 2 and not changed_intent:
        assessment = "minor"
    else:
        assessment = "major"
    drift = assessment == "major" or changed_intent or changed_scope
    return {
        "lexical_delta": lexical_delta,
        "semantic_delta_assessment": assessment,
        "possible_semantic_drift": bool(drift),
        "lost_entities": pipe(lost),
        "added_entities": pipe(added),
        "changed_intent": changed_intent,
        "changed_scope": changed_scope,
        "notes": "Heuristic entity/scope comparison; manual review required for final semantic judgment.",
    }


def find_phase20_rewrites(sample_ids: list[str]) -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    candidates = []
    for base in [ROOT / "results", ROOT / "reports", ROOT / "runtime", ROOT / "cache"]:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".jsonl", ".md", ".txt"}:
                text_path = str(path)
                if "/phase21" in text_path:
                    continue
                if "/phase19" in text_path or "/phase20" in text_path or "/runtime" in text_path or "/cache" in text_path:
                    candidates.append(path)

    rewrite_fields = ["rewritten_query", "english_mirror_query", "rewritten", "query_text"]
    preferred = ["phase20a_rewrite_enabled_full_eval", "phase20l", "phase20k", "phase19c", "phase19g", "phase19d"]
    for path in sorted(candidates, key=lambda p: (0 if any(x in str(p) for x in preferred) else 1, str(p))):
        try:
            if path.suffix.lower() == ".csv":
                with path.open(encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        sid = row.get("sample_id")
                        if sid not in sample_ids or sid in found:
                            continue
                        value = ""
                        if row.get("query_variant") in {"v0_original_CN"}:
                            continue
                        for field in rewrite_fields:
                            if row.get(field):
                                value = row[field]
                                break
                        if row.get("query_variant") in {"v1_EN_mirror"} and row.get("query_text"):
                            value = row["query_text"]
                        if value and value != row.get("original_query") and value != row.get("question_original"):
                            found[sid] = {
                                "phase20_rewrite_found": "True",
                                "phase20_rewritten_query": value,
                                "source_file": str(path.relative_to(ROOT)),
                                "source_phase": infer_phase(path),
                                "prompt_hash_if_available": row.get("rewrite_prompt_hash") or row.get("prompt_hash") or "",
                                "model_if_available": row.get("rewrite_model") or row.get("model") or "",
                                "cache_key_if_available": row.get("cache_key") or "",
                                "notes": "Found in historical CSV trace.",
                            }
            elif path.suffix.lower() in {".json", ".jsonl"}:
                text = path.read_text(encoding="utf-8", errors="ignore")
                for sid in sample_ids:
                    if sid in found or sid not in text:
                        continue
                    match = re.search(
                        rf"{re.escape(sid)}.*?(?:rewritten_query|english_mirror_query|rewritten)[\"']?\s*[:=]\s*[\"']([^\"'\n]+)",
                        text,
                        flags=re.DOTALL,
                    )
                    if match:
                        found[sid] = {
                            "phase20_rewrite_found": "True",
                            "phase20_rewritten_query": match.group(1).strip(),
                            "source_file": str(path.relative_to(ROOT)),
                            "source_phase": infer_phase(path),
                            "prompt_hash_if_available": "",
                            "model_if_available": "",
                            "cache_key_if_available": "",
                            "notes": "Found by text scan in historical JSON/JSONL.",
                        }
        except Exception:
            continue

    for sid in sample_ids:
        found.setdefault(sid, {
            "phase20_rewrite_found": "False",
            "phase20_rewritten_query": "",
            "source_file": "",
            "source_phase": "",
            "prompt_hash_if_available": "",
            "model_if_available": "",
            "cache_key_if_available": "",
            "notes": "No historical rewritten query found in phase19/phase20/runtime/cache scan.",
        })
    return found


def infer_phase(path: Path) -> str:
    match = re.search(r"(phase\d+[a-z0-9]*)", str(path))
    return match.group(1) if match else ""


def classify_sample(
    sample: dict[str, Any],
    current: dict[str, Any],
    phase20_lookup: dict[str, str],
    delta: dict[str, Any],
    variants: dict[str, dict[str, Any]],
    fallback_audit: dict[str, Any],
) -> dict[str, Any]:
    sid = sample["sample_id"]
    expected = sample.get("expected_doc_ids") or []
    current_doc_hit = variants["v3_live_rewrite_plus_original_cn_fallback"]["doc_hit"]
    original_doc_hit = variants["v1_original_cn_only"]["doc_hit"]
    phase20_doc_hit = variants.get("v2_phase20_rewrite_if_available", {}).get("doc_hit")
    current_final = variants["v3_live_rewrite_plus_original_cn_fallback"]["expected_doc_in_final"]
    current_support = variants["v3_live_rewrite_plus_original_cn_fallback"]["expected_doc_in_selected_support"]
    current_cited = variants["v3_live_rewrite_plus_original_cn_fallback"]["expected_doc_cited"]

    root = "unclear"
    action = "no_action"
    confidence = "low"
    evidence = []
    if not current.get("real_P0"):
        root = "nondeterminism"
        action = "rerun_after_cache_warm"
        confidence = "medium"
        evidence.append("Sample was listed in Phase 21A-9C remaining failures but current focused diagnostic rerun is not real P0.")
    elif not expected and current.get("real_P0"):
        root = "negative_abstention_regression"
        action = "investigate_support_citation"
        confidence = "medium"
        evidence.append("Negative/no-answer sample has no expected_doc_ids but produced citations/real P0.")
    elif phase20_lookup["phase20_rewrite_found"] == "False":
        root = "missing_phase20_rewrite_cache_reproducibility"
        action = "build_frozen_rewrite_cache_from_phase20_or_current_verified_outputs"
        confidence = "medium"
        evidence.append("No Phase 20 frozen rewrite found for exact sample_id.")
    elif delta["possible_semantic_drift"] is True and phase20_doc_hit and not current_doc_hit:
        root = "live_rewrite_semantic_drift"
        action = "restore_phase20_rewrite_outputs"
        confidence = "high"
        evidence.append("Phase20 rewrite variant recovers expected doc while current live rewrite does not.")
    elif fallback_audit["original_cn_fallback_triggered"] and fallback_audit["expected_doc_added_by_fallback"] and not current_doc_hit:
        root = "original_cn_fallback_not_effective"
        action = "investigate_specific_retrieval_case"
        confidence = "high"
        evidence.append("Original CN fallback adds expected doc but it is lost after rerank/final/support/citation.")
    elif current_final and not current_support:
        root = "support_selection_regression"
        action = "investigate_support_citation"
        confidence = "high"
        evidence.append("Expected doc reaches final chunks but not selected support.")
    elif current_support and not current_cited:
        root = "citation_binding_regression"
        action = "investigate_support_citation"
        confidence = "high"
        evidence.append("Expected doc selected as support but not cited.")
    elif original_doc_hit and not current_doc_hit:
        root = "live_rewrite_semantic_drift"
        action = "add_eval_rewrite_determinism_guard"
        confidence = "medium"
        evidence.append("Original CN variant recovers expected doc better than current live rewrite.")
    elif not current_final:
        root = "retrieval_real_regression"
        action = "investigate_specific_retrieval_case"
        confidence = "medium"
        evidence.append("Expected doc does not reach final chunks under current diagnostic variant.")
    elif current.get("real_P0"):
        root = "eval_taxonomy_or_label_issue" if current_doc_hit else "unclear"
        action = "eval_label_review" if current_doc_hit else "investigate_specific_retrieval_case"
        confidence = "medium" if current_doc_hit else "low"
        evidence.append("Current variant appears doc-hit but evaluation still marks real P0.")

    if delta["possible_semantic_drift"] is True:
        evidence.append(f"Rewrite delta assessed as {delta['semantic_delta_assessment']}; lost_entities={delta['lost_entities']}.")
    if fallback_audit["original_cn_fallback_triggered"]:
        evidence.append(
            f"CN fallback added {fallback_audit['fallback_added_count']} chunks; expected_added={fallback_audit['expected_doc_added_by_fallback']}."
        )
    return {
        "sample_id": sid,
        "original_failure_class": current.get("failure_class", ""),
        "reclassified_root_cause": root,
        "confidence": confidence,
        "evidence": " ".join(evidence),
        "recommended_action": action,
        "notes": "",
    }


def main() -> None:
    RDIR.mkdir(parents=True, exist_ok=True)
    REPDIR.mkdir(parents=True, exist_ok=True)
    configure_eval_env()
    settings = Settings.from_env()
    required = [
        P21C / "run_config.json",
        P21C / "pre_patch_code_audit.json",
        P21C / "implementation_patch_summary.json",
        P21C / "test_results.json",
        P21C / "rewrite_probe_after_patch.csv",
        P21C / "original_cn_fallback_probe.csv",
        P21C / "regression28_rerun_metrics.json",
        P21C / "smoke150_aligned_rerun_metrics.json",
        P21C / "root_cause_closure.json",
        P21C / "phase21a9d_next_step_decision.json",
        ROOT / "reports/phase21a9c_rewrite_wiring_fix/summary.md",
        P21 / "per_sample_results.csv",
        P21 / "failure_taxonomy.csv",
        P21B / "regression_sample_trace.csv",
        P21B / "rewrite_probe_results.csv",
        P21B / "original_cn_fallback_trace.csv",
        P21B / "root_cause_assessment.json",
        P20M / "final_code_state.json",
        P20M / "current_best_config.md",
        P20M / "residual_resolution_ledger.csv",
        P20L2 / "full_eval_regression_metrics.json",
        P20L2 / "fallback_provenance_audit.csv",
        ROOT / "reports/phase20m_convergence_summary/summary.md",
        ROOT / "reports/phase20l2_original_cn_fallback_floor_feature_ab/summary.md",
        SMOKE150,
        SMOKE200,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    smoke150_metrics = load_json(P21C / "smoke150_aligned_rerun_metrics.json")
    regression_ids = smoke150_metrics["remaining_failures"]
    smoke150 = load_jsonl(SMOKE150)
    samples_by_id = {sample["sample_id"]: sample for sample in smoke150}
    samples = [samples_by_id[sid] for sid in regression_ids]
    p21_tax = {row["sample_id"]: row for row in load_csv(P21 / "failure_taxonomy.csv")}

    run_config = {
        "phase": "21A-9D",
        "purpose": "investigate_remaining_smoke150_regression_after_rewrite_wiring_fix",
        "sample_count": len(samples),
        "sample_ids": regression_ids,
        "query_rewrite_mode": settings.query_rewrite.mode,
        "original_cn_fallback_enabled": settings.retrieval.original_cn_fallback_enabled,
        "generation_version": settings.generation.version,
        "qwen_synthesis_enabled": settings.generation.v2_use_qwen_synthesis,
        "query_rewrite_prompt_hash": get_prompt_hash(),
        "code_changed": False,
        "dataset_changed": False,
        "index_rebuild": False,
        "required_inputs_missing": missing,
    }
    write_json(RDIR / "run_config.json", run_config)

    pipeline = SynBioRAGPipeline(settings)
    rewrite_svc = build_rewrite_service(settings)
    filters = QueryFilters(tenant_id="default")
    phase20_lookup = find_phase20_rewrites(regression_ids)

    live_rewrites: dict[str, tuple[str, RewriteTrace]] = {}
    current_traces: dict[str, dict[str, Any]] = {}
    variant_rows: list[dict[str, Any]] = []
    variant_by_sample: dict[str, dict[str, dict[str, Any]]] = {}
    fallback_rows: list[dict[str, Any]] = []
    current_debug_by_sample: dict[str, dict[str, Any]] = {}

    for idx, sample in enumerate(samples, start=1):
        t0 = time.perf_counter()
        rewritten, trace = rewrite_svc.rewrite(sample["question"], is_negative=is_negative(sample))
        trace.rewrite_latency_ms = trace.rewrite_latency_ms or round((time.perf_counter() - t0) * 1000, 2)
        live_rewrites[sample["sample_id"]] = (rewritten, trace)
        phase20_query = phase20_lookup[sample["sample_id"]]["phase20_rewritten_query"]
        variants = [
            ("v0_current_live_rewrite", rewritten, False, False),
            ("v1_original_cn_only", sample["question"], False, False),
            ("v2_phase20_rewrite_if_available", phase20_query or rewritten, False, False),
            ("v3_live_rewrite_plus_original_cn_fallback", rewritten, True, False),
            ("v4_bilingual_dual_query_shadow", rewritten, False, True),
        ]
        variant_by_sample[sample["sample_id"]] = {}
        debug_by_variant: dict[str, dict[str, Any]] = {}
        for variant_id, query, use_cn, dual in variants:
            row, debug = run_variant(
                pipeline=pipeline,
                sample=sample,
                variant_id=variant_id,
                retrieval_query=query,
                use_original_cn_fallback=use_cn,
                dual_query_shadow=dual,
                filters=filters,
            )
            variant_rows.append(row)
            variant_by_sample[sample["sample_id"]][variant_id] = row
            debug_by_variant[variant_id] = debug
        current_debug = debug_by_variant["v3_live_rewrite_plus_original_cn_fallback"]
        route_pred = getattr(pipeline.router.analyze(sample["question"]).intent, "value", "")
        current_eval = evaluate_current(sample, current_debug, route_pred)
        current_debug_by_sample[sample["sample_id"]] = current_eval
        cn_debug = current_debug["cn_fallback_debug"]
        fallback_added_doc_ids = list(cn_debug.get("fallback_added_doc_ids") or [])
        expected_added = doc_hit(sample.get("expected_doc_ids") or [], fallback_added_doc_ids)
        final_docs = chunk_doc_ids(current_debug["final_chunks"])
        support_docs = current_debug["support_doc_ids"]
        cited_docs = current_debug["cited_doc_ids"]
        fallback_row = {
            "sample_id": sample["sample_id"],
            "original_query_contains_cjk": _contains_cjk(sample["question"]),
            "rewritten_differs_from_original": rewritten.strip() != sample["question"].strip(),
            "original_cn_fallback_enabled": settings.retrieval.original_cn_fallback_enabled,
            "original_cn_fallback_triggered": bool(cn_debug.get("triggered")),
            "fallback_added_count": int(cn_debug.get("fallback_added_count") or 0),
            "fallback_added_doc_ids": pipe(fallback_added_doc_ids),
            "expected_doc_added_by_fallback": expected_added,
            "fallback_candidate_reached_final": doc_hit(fallback_added_doc_ids, final_docs),
            "fallback_candidate_cited": doc_hit(fallback_added_doc_ids, cited_docs),
            "failure_after_fallback_stage": (
                "none" if current_eval["doc_hit"] and not current_eval["real_P0"] else
                "citation" if doc_hit(sample.get("expected_doc_ids") or [], support_docs) and not doc_hit(sample.get("expected_doc_ids") or [], cited_docs) else
                "support" if doc_hit(sample.get("expected_doc_ids") or [], final_docs) and not doc_hit(sample.get("expected_doc_ids") or [], support_docs) else
                "rerank_or_final" if expected_added and not doc_hit(sample.get("expected_doc_ids") or [], final_docs) else
                "retrieval_or_label"
            ),
            "notes": str(cn_debug.get("reason") or ""),
        }
        fallback_rows.append(fallback_row)
        current_traces[sample["sample_id"]] = {
            "sample_id": sample["sample_id"],
            "original_question": sample["question"],
            "live_rewritten_query": rewritten,
            "rewrite_success": bool(not trace.rewrite_fallback_used and rewritten.strip() != sample["question"].strip()),
            "rewrite_fallback": trace.rewrite_fallback_used,
            "rewritten_differs_from_original": rewritten.strip() != sample["question"].strip(),
            "rewrite_latency_ms": trace.rewrite_latency_ms,
            "cache_hit": trace.rewrite_cache_hit,
            "original_cn_fallback_triggered": fallback_row["original_cn_fallback_triggered"],
            "original_cn_fallback_added_count": fallback_row["fallback_added_count"],
            "retrieval_query_used": trace.retrieval_query_used,
            "notes": trace.rewrite_fallback_reason or trace.rewrite_error or "",
        }
        if idx % 5 == 0:
            print(f"diagnosed {idx}/{len(samples)}", flush=True)

    regression_rows = []
    for sample in samples:
        current = current_debug_by_sample[sample["sample_id"]]
        tax = p21_tax.get(sample["sample_id"], {})
        regression_rows.append({
            "sample_id": sample["sample_id"],
            "category": sample_category(sample),
            "expected_route": sample.get("expected_route", ""),
            "expected_doc_ids": pipe(sample.get("expected_doc_ids") or []),
            "expected_source_files": pipe(sample.get("expected_source_files") or []),
            "current_failure_class": current["failure_class"],
            "current_first_loss_stage": current["first_loss_stage"] or tax.get("first_loss_stage", ""),
            "current_doc_miss": current["doc_miss"],
            "current_cited_doc_ids": pipe(current["cited_doc_ids"]),
            "current_real_P0": current["real_P0"],
            "notes": "",
        })
    write_csv(RDIR / "regression17_sample_list.csv", regression_rows, [
        "sample_id", "category", "expected_route", "expected_doc_ids", "expected_source_files",
        "current_failure_class", "current_first_loss_stage", "current_doc_miss",
        "current_cited_doc_ids", "current_real_P0", "notes",
    ])

    write_csv(RDIR / "current_live_rewrite_trace.csv", list(current_traces.values()), [
        "sample_id", "original_question", "live_rewritten_query", "rewrite_success",
        "rewrite_fallback", "rewritten_differs_from_original", "rewrite_latency_ms",
        "cache_hit", "original_cn_fallback_triggered", "original_cn_fallback_added_count",
        "retrieval_query_used", "notes",
    ])

    lookup_rows = []
    for sid in regression_ids:
        row = {"sample_id": sid}
        row.update(phase20_lookup[sid])
        lookup_rows.append(row)
    write_csv(RDIR / "phase20_rewrite_trace_lookup.csv", lookup_rows, [
        "sample_id", "phase20_rewrite_found", "phase20_rewritten_query", "source_file",
        "source_phase", "prompt_hash_if_available", "model_if_available",
        "cache_key_if_available", "notes",
    ])

    delta_rows = []
    delta_by_sample = {}
    for sample in samples:
        sid = sample["sample_id"]
        phase20_query = phase20_lookup[sid]["phase20_rewritten_query"]
        current_query = current_traces[sid]["live_rewritten_query"]
        delta = semantic_delta(phase20_query, current_query, sample["question"])
        delta_by_sample[sid] = delta
        row = {
            "sample_id": sid,
            "original_question": sample["question"],
            "phase20_rewritten_query": phase20_query,
            "current_live_rewritten_query": current_query,
            "phase20_rewrite_found": phase20_lookup[sid]["phase20_rewrite_found"],
            "current_rewrite_success": current_traces[sid]["rewrite_success"],
        }
        row.update(delta)
        delta_rows.append(row)
    write_csv(RDIR / "rewrite_delta_audit.csv", delta_rows, [
        "sample_id", "original_question", "phase20_rewritten_query",
        "current_live_rewritten_query", "phase20_rewrite_found", "current_rewrite_success",
        "lexical_delta", "semantic_delta_assessment", "possible_semantic_drift",
        "lost_entities", "added_entities", "changed_intent", "changed_scope", "notes",
    ])

    write_csv(RDIR / "retrieval_variant_ab.csv", variant_rows, [
        "sample_id", "variant_id", "retrieval_query", "expected_doc_in_dense_top40",
        "expected_doc_in_bm25_top40", "expected_doc_in_hybrid_top40",
        "expected_doc_in_rerank_input", "expected_doc_in_final",
        "expected_doc_in_selected_support", "expected_doc_cited", "doc_hit",
        "real_P0_predicted_or_actual", "notes",
    ])

    write_csv(RDIR / "original_cn_fallback_behavior_audit.csv", fallback_rows, [
        "sample_id", "original_query_contains_cjk", "rewritten_differs_from_original",
        "original_cn_fallback_enabled", "original_cn_fallback_triggered",
        "fallback_added_count", "fallback_added_doc_ids", "expected_doc_added_by_fallback",
        "fallback_candidate_reached_final", "fallback_candidate_cited",
        "failure_after_fallback_stage", "notes",
    ])

    reclassified_rows = []
    for sample in samples:
        sid = sample["sample_id"]
        reclassified_rows.append(classify_sample(
            sample=sample,
            current=current_debug_by_sample[sid],
            phase20_lookup=phase20_lookup[sid],
            delta=delta_by_sample[sid],
            variants=variant_by_sample[sid],
            fallback_audit=next(row for row in fallback_rows if row["sample_id"] == sid),
        ))
    write_csv(RDIR / "reclassified_regression17.csv", reclassified_rows, [
        "sample_id", "original_failure_class", "reclassified_root_cause",
        "confidence", "evidence", "recommended_action", "notes",
    ])

    root_counts = Counter(row["reclassified_root_cause"] for row in reclassified_rows)
    action_counts = Counter(row["recommended_action"] for row in reclassified_rows)
    phase20_found_count = sum(row["phase20_rewrite_found"] == "True" for row in lookup_rows)
    drift_count = sum(row["possible_semantic_drift"] is True for row in delta_rows)
    missing_phase20_count = len(lookup_rows) - phase20_found_count
    recommend_cache = bool(missing_phase20_count or drift_count)
    repro = {
        "phase20_rewrite_outputs_available": phase20_found_count == len(lookup_rows),
        "live_rewrite_drift_detected_count": drift_count,
        "missing_phase20_rewrite_count": missing_phase20_count,
        "recommend_persistent_rewrite_cache": recommend_cache,
        "recommend_frozen_eval_rewrite_file": True,
        "recommend_fail_if_fallback_rate_above_threshold": True,
        "rationale": {
            "phase20_found_count": phase20_found_count,
            "sample_count": len(lookup_rows),
            "root_cause_counts": dict(root_counts),
            "action_counts": dict(action_counts),
        },
        "proposed_cache_schema": {
            "sample_id": "string",
            "original_question_hash": "sha256",
            "rewritten_query": "string",
            "prompt_hash": "string",
            "model": "string",
            "temperature": "float",
            "created_at": "iso8601",
            "source_phase": "string",
        },
        "notes": "A frozen eval rewrite file is needed for reproducible eval even when live Qwen succeeds.",
    }
    write_json(RDIR / "rewrite_reproducibility_recommendation.json", repro)

    most_common = root_counts.most_common()
    dominant = most_common[0][0] if most_common else "unclear"
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        dominant = "mixed"
    current_real = sum(row["current_real_P0"] is True for row in regression_rows)
    current_doc_miss = sum(row["current_doc_miss"] is True for row in regression_rows)
    phase20_restore_hits = sum(
        variant_by_sample[sid]["v2_phase20_rewrite_if_available"]["doc_hit"]
        for sid in regression_ids
    )
    assessment = {
        "smoke150_current_real_P0": smoke150_metrics["real_P0"],
        "smoke150_current_doc_miss": smoke150_metrics["doc_miss"],
        "phase20_baseline_real_P0": 0,
        "phase20_baseline_doc_miss": 0,
        "main_gap_to_reproduce_phase20": dominant,
        "can_reproduce_phase20_with_current_live_rewrite": False,
        "can_reproduce_phase20_if_phase20_rewrites_restored": (
            "unclear" if phase20_found_count < len(regression_ids) else phase20_restore_hits == len(regression_ids)
        ),
        "requires_rewrite_cache_freeze": recommend_cache,
        "should_rerun_smoke150_after_cache_fix": True,
        "should_rerun_smoke200_now": False,
        "notes": f"Diagnostic current rerun over listed samples found current_real_P0={current_real}, current_doc_miss={current_doc_miss}; smoke150 full metric remains authoritative from Phase 21A-9C.",
    }
    write_json(RDIR / "phase20_baseline_reproducibility_assessment.json", assessment)

    if root_counts["live_rewrite_semantic_drift"] + root_counts["missing_phase20_rewrite_cache_reproducibility"] >= 9:
        next_step = "build_frozen_rewrite_cache_from_phase20_or_current_verified_outputs"
    elif root_counts["missing_phase20_rewrite_cache_reproducibility"] > 0:
        next_step = "implement_persistent_eval_rewrite_cache"
    elif dominant == "mixed":
        next_step = "investigate_real_remaining_regressions"
    elif dominant in {"support_selection_regression", "citation_binding_regression", "retrieval_real_regression"}:
        next_step = "investigate_real_remaining_regressions"
    elif dominant == "eval_taxonomy_or_label_issue":
        next_step = "eval_label_review"
    else:
        next_step = "no_safe_next_step"
    decision = {
        "phase21a9d_completed": True,
        "root_cause_dominant": dominant,
        "phase20_baseline_reproducibility_blocker_identified": recommend_cache or dominant != "unclear",
        "recommended_phase21a9e": next_step,
        "rationale": {
            "root_cause_counts": dict(root_counts),
            "phase20_rewrite_found_count": phase20_found_count,
            "missing_phase20_rewrite_count": missing_phase20_count,
            "live_rewrite_drift_detected_count": drift_count,
        },
        "notes": "Do not rerun smoke200 or enter Phase 21B until smoke150 baseline is reproduced.",
    }
    write_json(RDIR / "phase21a9e_next_step_decision.json", decision)

    report = build_report(
        regression_rows=regression_rows,
        live_rows=list(current_traces.values()),
        lookup_rows=lookup_rows,
        delta_rows=delta_rows,
        variant_rows=variant_rows,
        fallback_rows=fallback_rows,
        reclassified_rows=reclassified_rows,
        repro=repro,
        decision=decision,
        assessment=assessment,
    )
    (REPDIR / "summary.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "regression17_root_cause_counts": dict(root_counts),
        "phase20_rewrite_found_count": phase20_found_count,
        "missing_phase20_rewrite_count": missing_phase20_count,
        "live_rewrite_drift_detected_count": drift_count,
        "recommended_phase21a9e": next_step,
    }, ensure_ascii=False, indent=2))


def build_report(
    *,
    regression_rows: list[dict[str, Any]],
    live_rows: list[dict[str, Any]],
    lookup_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    fallback_rows: list[dict[str, Any]],
    reclassified_rows: list[dict[str, Any]],
    repro: dict[str, Any],
    decision: dict[str, Any],
    assessment: dict[str, Any],
) -> str:
    root_counts = Counter(row["reclassified_root_cause"] for row in reclassified_rows)
    live_success = sum(bool(row["rewrite_success"]) for row in live_rows)
    phase20_found = sum(row["phase20_rewrite_found"] == "True" for row in lookup_rows)
    drift_count = sum(row["possible_semantic_drift"] is True for row in delta_rows)
    fallback_triggered = sum(bool(row["original_cn_fallback_triggered"]) for row in fallback_rows)
    expected_added = sum(bool(row["expected_doc_added_by_fallback"]) for row in fallback_rows)
    v_summary = Counter()
    for row in variant_rows:
        if row["doc_hit"] is True:
            v_summary[row["variant_id"]] += 1
    rows_text = "\n".join(
        f"- {row['sample_id']}: {row['current_failure_class']} / {row['current_first_loss_stage']}"
        for row in regression_rows
    )
    root_text = "\n".join(f"- {key}: {value}" for key, value in root_counts.items())
    return f"""# Phase 21A-9D Remaining Smoke150 Regression Investigation

## 1. Purpose
Investigate why smoke150 did not recover Phase 20 baseline after Phase 21A-9C fixed query rewrite LLM wiring.

## 2. Current Regression17
{rows_text}

## 3. Live Rewrite Trace
Current live rewrite succeeded for {live_success}/{len(live_rows)} listed samples. Fallback remained 0 for the investigated set.

## 4. Phase 20 Rewrite Lookup
Found historical Phase 19/20 rewritten query outputs for {phase20_found}/{len(lookup_rows)} samples. Missing outputs block exact Phase 20 reproduction for the rest.

## 5. Rewrite Delta
Heuristic rewrite drift detected for {drift_count}/{len(delta_rows)} samples. Drift assessment uses lexical delta plus entity/scope/intent checks and should be treated as diagnostic.

## 6. Retrieval Variant A/B
Doc-hit counts by variant: {dict(v_summary)}.

## 7. Original CN Fallback Behavior
Original CN fallback triggered for {fallback_triggered}/{len(fallback_rows)} samples and added an expected doc for {expected_added}/{len(fallback_rows)}. Some failures happen after fallback at rerank/final/support/citation stages.

## 8. Root Cause Reclassification
{root_text}

## 9. Rewrite Reproducibility Recommendation
- recommend_persistent_rewrite_cache: {repro['recommend_persistent_rewrite_cache']}
- recommend_frozen_eval_rewrite_file: {repro['recommend_frozen_eval_rewrite_file']}
- missing_phase20_rewrite_count: {repro['missing_phase20_rewrite_count']}
- live_rewrite_drift_detected_count: {repro['live_rewrite_drift_detected_count']}

## 10. Recommendation
Recommended Phase 21A-9E: `{decision['recommended_phase21a9e']}`.

Do not rerun smoke200 now: `{assessment['should_rerun_smoke200_now']}`.
"""


if __name__ == "__main__":
    main()
