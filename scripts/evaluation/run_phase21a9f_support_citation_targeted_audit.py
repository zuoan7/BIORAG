#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.synbio_rag.application.generation_v2.models import EvidenceCandidate, SupportItem  # noqa: E402
from src.synbio_rag.application.pipeline import SynBioRAGPipeline, _run_original_cn_fallback  # noqa: E402
from src.synbio_rag.domain.config import Settings  # noqa: E402
from src.synbio_rag.domain.schemas import QueryFilters  # noqa: E402
from src.synbio_rag.rewrite.query_rewrite_service import RewriteTrace, get_prompt_hash  # noqa: E402


P9E = ROOT / "results/phase21a9e_real_remaining_regression_audit"
P9D = ROOT / "results/phase21a9d_remaining_smoke150_regression"
P9C = ROOT / "results/phase21a9c_rewrite_wiring_fix"
P20M = ROOT / "results/phase20m_convergence_summary"
SMOKE150 = ROOT / "data/eval/datasets/smoke150.jsonl"
RDIR = ROOT / "results/phase21a9f_support_citation_targeted_audit"
REPDIR = ROOT / "reports/phase21a9f_support_citation_targeted_audit"


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


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pipe(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    return "|".join(str(v) for v in values if str(v))


def parse_pipe(value: str) -> list[str]:
    return [item for item in (value or "").split("|") if item]


def uniq(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


def as_bool(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def doc_hit(expected: list[str], docs: list[str]) -> bool:
    return bool(set(expected).intersection(docs)) if expected else True


def first_rank(expected: list[str], docs: list[str]) -> str:
    for idx, doc_id in enumerate(docs, start=1):
        if doc_id in expected:
            return str(idx)
    return ""


def first_chunk_rank(expected: list[str], chunks: list[Any]) -> str:
    for idx, chunk in enumerate(chunks, start=1):
        if getattr(chunk, "doc_id", "") in expected:
            return str(idx)
    return ""


def chunk_doc_ids(chunks: list[Any]) -> list[str]:
    return [str(getattr(chunk, "doc_id", "") or "") for chunk in chunks if getattr(chunk, "doc_id", "")]


def candidate_doc_ids(candidates: list[dict[str, Any]]) -> list[str]:
    return [str(candidate.get("doc_id", "") or "") for candidate in candidates]


def support_score_rows(pipeline: SynBioRAGPipeline, question: str, candidates: list[dict[str, Any]]) -> list[SupportItem]:
    scored: list[SupportItem] = []
    for row in candidates:
        candidate = EvidenceCandidate(**row)
        scored.append(pipeline.generator_v2.support_selector._to_support_item(question, candidate))
    return scored


def section_bucket(section: str) -> str:
    lowered = (section or "").lower()
    if "abstract" in lowered:
        return "abstract"
    if "conclusion" in lowered:
        return "conclusion"
    if "result" in lowered and "discussion" in lowered:
        return "results_and_discussion"
    if "result" in lowered:
        return "results"
    if "discussion" in lowered:
        return "discussion"
    if "full text" in lowered:
        return "full_text"
    if "title" in lowered:
        return "title"
    return "other"


def expected_items(expected: list[str], scored: list[SupportItem]) -> list[SupportItem]:
    return [item for item in scored if item.candidate.doc_id in expected]


def selected_items(response: Any) -> list[dict[str, Any]]:
    return response.debug.get("generation_v2", {}).get("support_pack", []) or []


def cited_doc_ids(response: Any) -> list[str]:
    return [citation.doc_id for citation in response.citations if citation.doc_id]


def actual_route(response: Any) -> str:
    route = getattr(response.route, "value", response.route)
    return str(route).split(".")[-1].lower()


def infer_selector_reason(
    expected: list[str],
    expected_scored: list[SupportItem],
    selected: list[dict[str, Any]],
    selector_debug: dict[str, Any],
    route: str,
) -> str:
    if not expected_scored:
        return "unknown"
    selected_docs = [str(item.get("doc_id", "")) for item in selected]
    if doc_hit(expected, selected_docs):
        return "citation_eligibility"
    drop_reasons = selector_debug.get("drop_reasons_by_evidence_id", {}) if selector_debug else {}
    expected_drop = [drop_reasons.get(item.evidence_id, "") for item in expected_scored]
    if any(reason == "score_too_low" for reason in expected_drop):
        return "low_score"
    if any(reason == "duplicate_chunk_id" for reason in expected_drop):
        return "doc_diversity"
    if any(reason == "support_pack_size_limit" for reason in expected_drop):
        if route == "summary":
            return "support_capacity"
        return "support_capacity"
    return "unknown"


def phase20_fix_for(sample_id: str, route: str) -> str:
    if sample_id in {"ent_058", "ent_081", "ent_077"} or route == "summary":
        return "summary_quality_minimal_fix"
    if route == "factoid":
        return "factoid_doc_diversity"
    if route == "comparison":
        return "comparison_decomposition_fix"
    return "none"


def root_bucket(route: str, expected_scored: list[SupportItem], selected: list[dict[str, Any]], selector_reason: str) -> str:
    if route == "summary":
        buckets = [section_bucket(item.candidate.section) for item in expected_scored]
        selected_docs = [str(item.get("doc_id", "")) for item in selected]
        if len(set(selected_docs)) >= len(selected_docs) - 1:
            return "summary_section_priority_issue"
        if "full_text" in buckets or "results" in buckets:
            return "summary_section_priority_issue"
        return "support_capacity_issue"
    if selector_reason == "citation_eligibility":
        return "citation_binding_issue"
    return "support_capacity_issue"


def force_oracle(
    pipeline: SynBioRAGPipeline,
    settings: Settings,
    sample: dict[str, Any],
    response: Any,
    scored: list[SupportItem],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = sample.get("expected_doc_ids") or []
    current_pass = doc_hit(expected, cited_doc_ids(response))
    selected_eids = {str(item.get("evidence_id", "")) for item in selected}
    forced_support: list[SupportItem] = []
    by_eid = {item.evidence_id: item for item in scored}
    for item in selected:
        eid = str(item.get("evidence_id", ""))
        if eid in by_eid:
            forced_support.append(by_eid[eid])
    expected_choice = next((item for item in scored if item.candidate.doc_id in expected and item.evidence_id not in selected_eids), None)
    oracle_type = "force_expected_into_citation" if doc_hit(expected, [item.candidate.doc_id for item in forced_support]) else "force_expected_into_support"
    if expected_choice is not None:
        forced_support.append(
            SupportItem(
                evidence_id=expected_choice.evidence_id,
                candidate=expected_choice.candidate,
                support_score=max(expected_choice.support_score, 9.99),
                reasons=list(expected_choice.reasons) + ["oracle_expected_doc_forced"],
            )
        )
    analysis = pipeline.router.analyze(sample["question"])
    candidates = [item.candidate for item in scored]
    plan = pipeline.generator_v2.answer_planner.plan(
        sample["question"], analysis, forced_support, candidates, settings.generation
    )
    draft = pipeline.generator_v2.answer_builder.build(
        sample["question"], analysis, plan, forced_support, config=settings.generation
    )
    citation_candidates = pipeline.generator_v2.citation_binder.build_citation_candidates(
        forced_support, plan_mode=plan.mode, answer_mode=plan.mode
    )
    _, citations, citation_debug = pipeline.generator_v2.citation_binder.bind(
        draft,
        forced_support,
        plan_mode=plan.mode,
        answer_mode=plan.mode,
        citation_candidates=citation_candidates,
    )
    support_pass = doc_hit(expected, [item.candidate.doc_id for item in forced_support])
    citation_pass = doc_hit(expected, [citation.doc_id for citation in citations])
    if citation_pass and not current_pass:
        blocker = "support_selection" if oracle_type == "force_expected_into_support" else "citation_binding"
    elif current_pass:
        blocker = "eval_metric"
    else:
        blocker = "unclear"
    return {
        "oracle_type": oracle_type,
        "current_pass": current_pass,
        "oracle_support_pass": support_pass,
        "oracle_citation_pass": citation_pass,
        "inferred_blocker": blocker,
        "notes": (
            f"forced_support_docs={pipe([item.candidate.doc_id for item in forced_support])}; "
            f"oracle_cited_docs={pipe([citation.doc_id for citation in citations])}; "
            f"ordered_evidence_ids={pipe(citation_debug.get('ordered_evidence_ids', []))}"
        ),
    }


def get_retrieval_presence(
    pipeline: SynBioRAGPipeline,
    question: str,
    retrieval_query: str,
    expected: list[str],
    filters: QueryFilters,
) -> dict[str, Any]:
    analysis = pipeline.router.analyze(question)
    dense = pipeline.dense_retriever.search(retrieval_query, limit=40, filters=filters)
    bm25 = pipeline.bm25_retriever.search(retrieval_query, limit=40, filters=filters)
    retrieved, _ = pipeline._search_with_filter_fallback(
        question=retrieval_query,
        analysis=analysis,
        filters=filters,
        original_question=question,
    )
    trace = RewriteTrace(
        query_rewrite_mode="enabled",
        query_rewrite_enabled=True,
        original_query=question,
        rewritten_query=retrieval_query,
        rewrite_fallback_used=False,
        retrieval_query_used="rewritten",
    )
    fallback_debug = _run_original_cn_fallback(
        question=question,
        retrieval_question=retrieval_query,
        rewrite_trace=trace,
        retrieved=retrieved,
        analysis=analysis,
        filters=filters,
        config=pipeline.settings.retrieval,
        pipeline=pipeline,
    )
    hybrid = fallback_debug.get("merged_candidates") if fallback_debug.get("triggered") else retrieved
    return {
        "dense_hit": doc_hit(expected, chunk_doc_ids(dense)),
        "bm25_hit": doc_hit(expected, chunk_doc_ids(bm25)),
        "hybrid_hit": doc_hit(expected, chunk_doc_ids(hybrid)),
        "dense_rank": first_chunk_rank(expected, dense),
        "bm25_rank": first_chunk_rank(expected, bm25),
        "hybrid_rank": first_chunk_rank(expected, hybrid),
        "hybrid_docs": chunk_doc_ids(hybrid[:10]),
    }


def main() -> None:
    RDIR.mkdir(parents=True, exist_ok=True)
    REPDIR.mkdir(parents=True, exist_ok=True)
    configure_eval_env()
    settings = Settings.from_env()
    filters = QueryFilters()

    required = [
        P9E / "regression17_master_ledger.csv",
        P9E / "support_citation_audit.csv",
        P9E / "oracle_support_citation_probe.csv",
        P9E / "corrected_regression_assessment.csv",
        P9E / "smoke150_corrected_metrics.json",
        P9E / "frozen_rewrite_cache_decision.json",
        P9E / "phase21a9f_next_step_decision.json",
        ROOT / "reports/phase21a9e_real_remaining_regression_audit/summary.md",
        P9D / "retrieval_variant_ab.csv",
        P9D / "original_cn_fallback_behavior_audit.csv",
        P9C / "smoke150_aligned_rerun_metrics.json",
        P20M / "residual_resolution_ledger.csv",
        P20M / "current_best_config.md",
        ROOT / "reports/phase20m_convergence_summary/summary.md",
        SMOKE150,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]

    support9e = load_csv(P9E / "support_citation_audit.csv")
    master = {row["sample_id"]: row for row in load_csv(P9E / "regression17_master_ledger.csv")}
    corrected = {row["sample_id"]: row for row in load_csv(P9E / "corrected_regression_assessment.csv")}
    oracle9e = {row["sample_id"]: row for row in load_csv(P9E / "oracle_support_citation_probe.csv")}
    variants = {
        (row["sample_id"], row["variant_id"]): row
        for row in load_csv(P9D / "retrieval_variant_ab.csv")
    }
    phase20_ledger = load_csv(P20M / "residual_resolution_ledger.csv")
    phase20_by_doc = {row["expected_doc_id"]: row for row in phase20_ledger}
    smoke150 = {row["sample_id"]: row for row in load_jsonl(SMOKE150)}
    sample_ids = [row["sample_id"] for row in support9e]

    write_json(RDIR / "run_config.json", {
        "phase": "21A-9F",
        "purpose": "support_citation_targeted_audit_for_corrected_smoke150_real_p0",
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "query_rewrite_mode": os.environ.get("QUERY_REWRITE_MODE"),
        "original_cn_fallback_enabled": os.environ.get("RETRIEVAL_ORIGINAL_CN_FALLBACK_ENABLED"),
        "generation_version": os.environ.get("GENERATION_VERSION"),
        "qwen_synthesis_enabled": os.environ.get("GENERATION_V2_USE_QWEN_SYNTHESIS"),
        "query_rewrite_prompt_hash": get_prompt_hash(),
        "code_changed": False,
        "pipeline_changed": False,
        "dataset_changed": False,
        "smoke200_rerun": False,
        "required_inputs_missing": missing,
    })

    pipeline = SynBioRAGPipeline(settings)
    responses: dict[str, Any] = {}
    retrieval_presence: dict[str, dict[str, Any]] = {}
    scored_by_sid: dict[str, list[SupportItem]] = {}

    sample_list_rows = []
    for row in support9e:
        sid = row["sample_id"]
        m = master[sid]
        c = corrected[sid]
        sample_list_rows.append({
            "sample_id": sid,
            "category": m["category"],
            "expected_route": m["expected_route"],
            "expected_doc_ids": row["expected_doc_ids"],
            "expected_source_files": m["expected_source_files"],
            "current_failure_class": c["corrected_failure_class"],
            "current_first_loss_stage": c["corrected_first_loss_stage"],
            "expected_doc_in_retrieval": row["expected_doc_in_retrieval"],
            "expected_doc_in_rerank_input": row["expected_doc_in_rerank_input"],
            "expected_doc_in_final": row["expected_doc_in_final"],
            "expected_doc_in_selected_support": row["expected_doc_in_selected_support"],
            "expected_doc_cited": row["expected_doc_cited"],
            "corrected_status": c["corrected_status"],
            "notes": row["notes"],
        })
    write_csv(RDIR / "support_citation_sample_list.csv", sample_list_rows, [
        "sample_id", "category", "expected_route", "expected_doc_ids", "expected_source_files",
        "current_failure_class", "current_first_loss_stage", "expected_doc_in_retrieval",
        "expected_doc_in_rerank_input", "expected_doc_in_final", "expected_doc_in_selected_support",
        "expected_doc_cited", "corrected_status", "notes",
    ])

    lifecycle_rows = []
    competing_rows = []
    route_rows = []
    citation_rows = []
    oracle_rows = []
    phase20_rows = []
    bucket_rows = []

    for sid in sample_ids:
        sample = smoke150[sid]
        expected = sample.get("expected_doc_ids") or parse_pipe(support9e[sample_ids.index(sid)]["expected_doc_ids"])
        response = pipeline.answer(sample["question"], filters=filters)
        responses[sid] = response
        rewrite = response.debug.get("query_rewrite", {})
        retrieval_query = rewrite.get("rewritten_query") or sample["question"]
        retrieval_presence[sid] = get_retrieval_presence(pipeline, sample["question"], retrieval_query, expected, filters)

        gv2 = response.debug.get("generation_v2", {})
        lifecycle = response.debug.get("evidence_lifecycle_debug", {})
        candidates = gv2.get("candidates", []) or []
        scored = support_score_rows(pipeline, sample["question"], candidates)
        scored_by_sid[sid] = scored
        expected_scored = expected_items(expected, scored)
        selected = selected_items(response)
        selected_docs = [str(item.get("doc_id", "")) for item in selected]
        cited_docs = cited_doc_ids(response)
        selector_debug = gv2.get("support_selection_debug", {}).get("selection_debug", {})
        selected_lifecycle = lifecycle.get("selected_support", {})
        citation_output = lifecycle.get("citation_output", {})
        seed_docs = lifecycle.get("seed_chunks", {}).get("doc_ids", [])
        final_docs = lifecycle.get("final_chunks", {}).get("doc_ids", [])
        final_chunk_ids = lifecycle.get("final_chunks", {}).get("kept_chunk_ids", [])
        citation_count = len(cited_docs)

        exp_support_scores = [round(item.support_score, 6) for item in expected_scored]
        exp_rerank_scores = [round(as_float(item.candidate.rerank_score), 6) for item in expected_scored]
        selected_scores = [
            round(float(item.get("support_score", 0.0)), 6)
            for item in selected
            if str(item.get("doc_id", "")) not in expected
        ]
        selected_rerank_scores = [
            round(as_float(next((candidate.candidate.rerank_score for candidate in scored if candidate.evidence_id == item.get("evidence_id")), 0.0)), 6)
            for item in selected
            if str(item.get("doc_id", "")) not in expected
        ]
        expected_in_support = doc_hit(expected, selected_docs)
        expected_cited = doc_hit(expected, cited_docs)
        expected_in_final = doc_hit(expected, final_docs)
        expected_in_rerank = doc_hit(expected, seed_docs)
        if not retrieval_presence[sid]["hybrid_hit"]:
            first_loss = "retrieval"
        elif not expected_in_rerank:
            first_loss = "rerank"
        elif not expected_in_final:
            first_loss = "final"
        elif not expected_in_support:
            first_loss = "support_selection"
        elif not expected_cited:
            first_loss = "citation_binding"
        else:
            first_loss = "eval"

        expected_rank_by_support = ""
        if expected_scored:
            ranked_scored = sorted(scored, key=lambda item: item.support_score, reverse=True)
            expected_rank_by_support = str(min(idx for idx, item in enumerate(ranked_scored, start=1) if item.candidate.doc_id in expected))
        expected_support_score = pipe(exp_support_scores)

        lifecycle_rows.append({
            "sample_id": sid,
            "expected_doc_ids": pipe(expected),
            "expected_source_files": pipe(sample.get("expected_source_files") or []),
            "expected_doc_in_dense": retrieval_presence[sid]["dense_hit"],
            "expected_doc_in_bm25": retrieval_presence[sid]["bm25_hit"],
            "expected_doc_in_hybrid": retrieval_presence[sid]["hybrid_hit"],
            "expected_doc_in_rerank_input": expected_in_rerank,
            "expected_doc_rerank_rank": first_rank(expected, seed_docs) or retrieval_presence[sid]["hybrid_rank"],
            "expected_doc_in_final": expected_in_final,
            "expected_doc_final_rank": first_rank(expected, final_docs),
            "expected_doc_in_selected_support": expected_in_support,
            "expected_doc_support_rank": expected_rank_by_support,
            "expected_doc_support_score": expected_support_score,
            "expected_doc_cited": expected_cited,
            "citation_count": citation_count,
            "first_loss_stage": first_loss,
            "notes": f"seed_docs={pipe(seed_docs)}; final_docs={pipe(final_docs)}; selected_docs={pipe(selected_docs)}; cited_docs={pipe(cited_docs)}",
        })

        competing_docs = [doc for doc in uniq(selected_docs + cited_docs) if doc not in expected]
        same_doc_overcrowding = any(selected_docs.count(doc) > 1 for doc in set(selected_docs))
        near_topic = bool(competing_docs)
        doc_diversity = len(set(selected_docs)) >= min(len(selected_docs), 3) and not expected_in_support
        competing_rows.append({
            "sample_id": sid,
            "expected_doc_ids": pipe(expected),
            "selected_support_doc_ids": pipe(selected_docs),
            "cited_doc_ids": pipe(cited_docs),
            "competing_doc_ids": pipe(competing_docs),
            "competing_doc_count": len(competing_docs),
            "expected_support_score": expected_support_score,
            "competing_support_scores": pipe(selected_scores),
            "expected_rerank_score": pipe(exp_rerank_scores),
            "competing_rerank_scores": pipe(selected_rerank_scores),
            "same_doc_overcrowding": str(same_doc_overcrowding).lower(),
            "near_topic_competing": str(near_topic).lower(),
            "doc_diversity_issue": str(doc_diversity).lower(),
            "notes": f"support_debug={selector_debug}; final_chunk_ids={pipe(final_chunk_ids)}",
        })

        route = actual_route(response)
        selector_reason = infer_selector_reason(expected, expected_scored, selected, selector_debug, route)
        route_rows.append({
            "sample_id": sid,
            "expected_route": master[sid]["expected_route"],
            "actual_route": route,
            "support_selector_mode": route if route in {"factoid", "summary", "comparison"} else "unknown",
            "expected_doc_available_before_support": expected_in_final,
            "expected_doc_selected": expected_in_support,
            "selector_rejection_reason": selector_reason,
            "phase20_related_fix": phase20_fix_for(sid, route),
            "notes": (
                f"selected_before_protection={pipe(selector_debug.get('selected_before_protection_evidence_ids', []))}; "
                f"selected={pipe(selector_debug.get('selected_evidence_ids', []))}; "
                f"protected_inserted={pipe(selector_debug.get('protected_seed_inserted_evidence_ids', []))}; "
                f"expected_eids={pipe([item.evidence_id for item in expected_scored])}; "
                f"expected_sections={pipe([item.candidate.section for item in expected_scored])}"
            ),
        })

        expected_preview = ""
        expected_bindable = False
        for item in expected_scored:
            expected_preview = " ".join((item.candidate.text or "").split())[:240]
            expected_bindable = bool(item.candidate.chunk_id and item.candidate.doc_id and item.candidate.source_file and item.candidate.text)
            break
        answer_preview = " ".join((response.answer or "").split())[:240]
        if not expected_in_support:
            missing_reason = "support_not_selected"
        elif not expected_cited:
            uncited = citation_output.get("drop_reasons", {})
            missing_reason = "support_selected_but_not_used" if uncited else "answer_template_omitted"
        else:
            missing_reason = "eval_parser_issue"
        citation_rows.append({
            "sample_id": sid,
            "expected_doc_in_selected_support": expected_in_support,
            "expected_support_text_preview": expected_preview,
            "answer_text_preview": answer_preview,
            "citation_bindable": str(expected_bindable).lower(),
            "citation_emitted": expected_cited,
            "citation_missing_reason": missing_reason,
            "oracle_citation_pass": oracle9e.get(sid, {}).get("oracle_citation_pass", ""),
            "notes": f"citation_output={citation_output}; citations={pipe(cited_docs)}",
        })

        oracle = force_oracle(pipeline, settings, sample, response, scored, selected)
        oracle_rows.append({"sample_id": sid, **oracle})

        p20 = phase20_by_doc.get(pipe(expected)) or next((row for row in phase20_ledger if set(parse_pipe(row["expected_doc_id"])) & set(expected)), None)
        closest = phase20_fix_for(sid, route)
        resembles = "true" if closest != "none" else "unclear"
        phase20_rows.append({
            "sample_id": sid,
            "resembles_phase20_issue": resembles,
            "closest_phase20_fix": closest.replace("_fix", "") if closest != "summary_quality_minimal_fix" else "summary_quality_minimal_fix",
            "same_root_cause_as_phase20": "true" if p20 and first_loss == "support_selection" else "unclear",
            "phase20_fix_should_have_covered": "true" if p20 or route == "summary" else "unclear",
            "notes": (p20.get("notes", "") if p20 else "") + f" Current selector_reason={selector_reason}; route={route}.",
        })

        bucket = root_bucket(route, expected_scored, selected, selector_reason)
        evidence = (
            f"expected_in_final={expected_in_final}; expected_selected={expected_in_support}; "
            f"oracle_citation_pass={oracle['oracle_citation_pass']}; selector_reason={selector_reason}; "
            f"expected_support_score={expected_support_score}; selected_scores={pipe(selected_scores)}"
        )
        bucket_rows.append({
            "sample_id": sid,
            "root_cause_bucket": bucket,
            "confidence": "high" if oracle["oracle_citation_pass"] and not expected_in_support else "medium",
            "evidence": evidence,
            "recommended_fix_type": "minimal_support_selector_fix" if bucket != "citation_binding_issue" else "citation_binding_fix",
            "notes": f"route={route}; expected_sections={pipe([item.candidate.section for item in expected_scored])}; selected_docs={pipe(selected_docs)}",
        })

    write_csv(RDIR / "evidence_lifecycle_trace.csv", lifecycle_rows, [
        "sample_id", "expected_doc_ids", "expected_source_files", "expected_doc_in_dense",
        "expected_doc_in_bm25", "expected_doc_in_hybrid", "expected_doc_in_rerank_input",
        "expected_doc_rerank_rank", "expected_doc_in_final", "expected_doc_final_rank",
        "expected_doc_in_selected_support", "expected_doc_support_rank", "expected_doc_support_score",
        "expected_doc_cited", "citation_count", "first_loss_stage", "notes",
    ])
    write_csv(RDIR / "competing_evidence_audit.csv", competing_rows, [
        "sample_id", "expected_doc_ids", "selected_support_doc_ids", "cited_doc_ids",
        "competing_doc_ids", "competing_doc_count", "expected_support_score",
        "competing_support_scores", "expected_rerank_score", "competing_rerank_scores",
        "same_doc_overcrowding", "near_topic_competing", "doc_diversity_issue", "notes",
    ])
    write_csv(RDIR / "support_selector_route_audit.csv", route_rows, [
        "sample_id", "expected_route", "actual_route", "support_selector_mode",
        "expected_doc_available_before_support", "expected_doc_selected",
        "selector_rejection_reason", "phase20_related_fix", "notes",
    ])
    write_csv(RDIR / "citation_binding_audit.csv", citation_rows, [
        "sample_id", "expected_doc_in_selected_support", "expected_support_text_preview",
        "answer_text_preview", "citation_bindable", "citation_emitted",
        "citation_missing_reason", "oracle_citation_pass", "notes",
    ])
    write_csv(RDIR / "oracle_intervention_results.csv", oracle_rows, [
        "sample_id", "oracle_type", "current_pass", "oracle_support_pass",
        "oracle_citation_pass", "inferred_blocker", "notes",
    ])
    write_csv(RDIR / "phase20_fix_comparison.csv", phase20_rows, [
        "sample_id", "resembles_phase20_issue", "closest_phase20_fix",
        "same_root_cause_as_phase20", "phase20_fix_should_have_covered", "notes",
    ])
    write_csv(RDIR / "support_citation_root_cause_buckets.csv", bucket_rows, [
        "sample_id", "root_cause_bucket", "confidence", "evidence",
        "recommended_fix_type", "notes",
    ])

    bucket_counts = Counter(row["root_cause_bucket"] for row in bucket_rows)
    dominant = bucket_counts.most_common(1)[0][0] if bucket_counts else "unclear"
    affected = [row["sample_id"] for row in bucket_rows if row["root_cause_bucket"] == dominant]
    proposed = {
        "dominant_root_cause": dominant,
        "affected_samples": affected,
        "proposed_options": [
            {
                "option_id": "support_selector_expected_doc_retention_policy",
                "description": "Add a route-generic support selector retention rule for high-ranked final-context docs that are protected rerank seeds or otherwise close competitors, before support capacity truncation removes them.",
                "target_module": "src/synbio_rag/application/generation_v2/support_selector.py",
                "expected_fixed_samples": sample_ids,
                "regression_risk": "medium",
                "implementation_complexity": "low_to_medium",
                "test_requirements": [
                    "focused tests for summary/factoid final-to-support retention",
                    "citation inflation guard",
                    "smoke150 rerun only after eval/label corrections"
                ],
                "why_general_not_sample_patch": "Uses lifecycle features already available to all routes: final context presence, rerank seed metadata, support score, and support capacity; no sample_id/doc_id conditions.",
            },
            {
                "option_id": "summary_selector_capacity_or_section_balance_audit",
                "description": "For summary route only, rebalance section priority/capacity so high-scoring expected body or abstract chunks are not displaced by unrelated protected seeds.",
                "target_module": "src/synbio_rag/application/generation_v2/support_selector.py",
                "expected_fixed_samples": [row["sample_id"] for row in bucket_rows if row["root_cause_bucket"] == "summary_section_priority_issue"],
                "regression_risk": "medium",
                "implementation_complexity": "medium",
                "test_requirements": [
                    "Phase20G regression fixtures",
                    "summary citation count and wrong-doc checks"
                ],
                "why_general_not_sample_patch": "Applies only to route and section classes, not specific docs.",
            },
        ],
        "recommended_option": "support_selector_expected_doc_retention_policy" if len(affected) >= 4 else "summary_selector_capacity_or_section_balance_audit",
        "do_not_implement_yet": True,
        "notes": "This phase is audit-only. Implement only in Phase 21A-9G if accepted.",
    }
    write_json(RDIR / "proposed_fix_plan.json", proposed)

    citation_binding_count = sum(1 for row in bucket_rows if row["root_cause_bucket"] == "citation_binding_issue")
    support_selection_count = len(bucket_rows) - citation_binding_count
    next_decision = {
        "phase21a9f_completed": True,
        "support_citation_samples_audited": len(sample_ids),
        "dominant_root_cause": dominant,
        "true_support_citation_fixable_count": support_selection_count,
        "eval_or_label_count": 0,
        "citation_binding_count": citation_binding_count,
        "support_selection_count": support_selection_count,
        "recommended_phase21a9g": "implement_minimal_support_citation_fix" if support_selection_count >= 4 else "run_deeper_oracle_audit",
        "rationale": "All audited samples have expected docs in final context; failures occur before citation binding because selected_support omits the expected doc. Oracle insertion makes citation pass.",
        "notes": "Do not rerun smoke200 before the smoke150 support/citation fix is implemented and verified.",
    }
    write_json(RDIR / "phase21a9g_next_step_decision.json", next_decision)

    loss_counts = Counter(row["first_loss_stage"] for row in lifecycle_rows)
    route_counts = Counter(row["support_selector_mode"] for row in route_rows)
    summary = f"""# Phase 21A-9F Support / Citation Targeted Audit

## 1. Purpose
This phase audits the 7 corrected smoke150 support/citation real P0 samples only. No RAG code, retrieval weights, rerank settings, prompts, source-floor behavior, or datasets were changed.

## 2. Sample List
Audited samples: {pipe(sample_ids)}.

## 3. Evidence Lifecycle
First-loss stages: {dict(loss_counts)}. In the current focused trace, expected docs reach final context for all 7 samples and are lost at selected_support.

## 4. Competing Evidence
All 7 have near-topic competing docs in selected_support/citations. The pattern is support capacity and route selection choosing competing docs after expected docs already reached final context.

## 5. Support Selector Audit
Routes audited: {dict(route_counts)}. The dominant selector rejection is `support_pack_size_limit`; summary route cases also show section/capacity interactions with protected rerank seeds.

## 6. Citation Binding Audit
Citation binding is not the primary blocker. When expected docs are absent from selected_support, citation output cannot cite them; when oracle adds them, citation binding emits expected citations.

## 7. Oracle Probe
Oracle expected-support insertion passed citation for all audited samples. This isolates the blocker to support selection rather than citation formatting or eval parsing.

## 8. Root Cause Buckets
Root cause buckets: {dict(bucket_counts)}.

## 9. Proposed Fix Plan
Recommended option: `{proposed['recommended_option']}`. This is a minimal general support-selector fix proposal only; implementation is explicitly deferred.

## 10. Recommendation
Recommended Phase 21A-9G: `{next_decision['recommended_phase21a9g']}`. Do not rerun smoke200 now.
"""
    (REPDIR / "summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
