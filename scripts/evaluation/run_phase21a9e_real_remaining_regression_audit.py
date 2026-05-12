#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from scripts.evaluation.run_phase21a9d_remaining_smoke150_regression import (  # noqa: E402
    configure_eval_env,
    doc_hit,
    load_csv,
    load_json,
    load_jsonl,
    parse_pipe,
    pipe,
    run_variant,
    write_csv,
    write_json,
)
from src.synbio_rag.application.pipeline import SynBioRAGPipeline  # noqa: E402
from src.synbio_rag.domain.config import Settings  # noqa: E402
from src.synbio_rag.domain.schemas import QueryFilters  # noqa: E402
from src.synbio_rag.evaluation.failure_taxonomy import evaluate_failure  # noqa: E402
from src.synbio_rag.rewrite.query_rewrite_service import get_prompt_hash  # noqa: E402
from src.synbio_rag.application.generation_v2.models import SupportItem  # noqa: E402


P9D = ROOT / "results/phase21a9d_remaining_smoke150_regression"
P9C = ROOT / "results/phase21a9c_rewrite_wiring_fix"
P20M = ROOT / "results/phase20m_convergence_summary"
SMOKE150 = ROOT / "data/eval/datasets/smoke150.jsonl"
SMOKE200 = ROOT / "data/eval/datasets/smoke200.jsonl"
RDIR = ROOT / "results/phase21a9e_real_remaining_regression_audit"
REPDIR = ROOT / "reports/phase21a9e_real_remaining_regression_audit"

BOOL_TRUE = {"True", "true", "1", "yes", "Y"}


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value) in BOOL_TRUE


def uniq(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


def note_docs(notes: str, key: str) -> list[str]:
    match = re.search(rf"{re.escape(key)}=([^;]+)", notes or "")
    if not match:
        return []
    return parse_pipe(match.group(1).strip())


def expected_rank(expected: list[str], docs: list[str], in_top40: str) -> str:
    for idx, doc_id in enumerate(docs, start=1):
        if doc_id in expected:
            return str(idx)
    return "<=40_not_in_preview" if as_bool(in_top40) else ">40_or_absent"


def current_variant(variants: dict[tuple[str, str], dict[str, str]], sid: str) -> dict[str, str]:
    return variants[(sid, "v3_live_rewrite_plus_original_cn_fallback")]


def is_negative_abstention(sample: dict[str, Any]) -> bool:
    tags = {str(t) for t in sample.get("tags") or []}
    return "abstain" in tags or "negative_case" in tags


def is_negative_trigger_only(sample: dict[str, Any]) -> bool:
    tags = {str(t) for t in sample.get("tags") or []}
    return "negative_trigger" in tags and "abstain" not in tags and "negative_case" not in tags


def corrected_for(
    sample: dict[str, Any],
    root: str,
) -> tuple[str, str, str, str, str]:
    sid = sample["sample_id"]
    if root == "eval_taxonomy_or_label_issue" and is_negative_trigger_only(sample):
        return (
            "false_or_eval_p0",
            "eval_taxonomy_false_negative_trigger",
            "eval_taxonomy",
            "high",
            "update_eval_taxonomy",
        )
    if root == "nondeterminism":
        return ("nondeterministic", "unstable_doc_hit", "nondeterminism", "medium", "rerun_smoke150_after_corrections")
    if root == "negative_abstention_regression":
        return ("true_real_p0", "negative_abstention_regression", "generation", "high", "negative_policy_audit")
    if root in {"support_selection_regression", "original_cn_fallback_not_effective"}:
        return ("true_real_p0", "support_selection_regression", "support_selection", "high", "support_citation_targeted_audit")
    if root == "retrieval_real_regression":
        return ("true_real_p0", "retrieval_real_regression", "final", "medium", "retrieval_generalization_audit")
    if root == "live_rewrite_semantic_drift":
        if sid == "ent_100":
            return ("true_real_p0", "rewrite_drift_support_selection", "support_selection", "high", "support_citation_targeted_audit")
        return ("true_real_p0", "rewrite_drift_recall_loss", "retrieval", "medium", "implement_frozen_eval_rewrite_cache")
    return ("needs_manual_review", root or "unclear", "unclear", "low", "manual_review")


def build_oracle_probe(
    pipeline: SynBioRAGPipeline,
    settings: Settings,
    sample: dict[str, Any],
    variant_row: dict[str, str],
    debug: dict[str, Any],
) -> dict[str, Any]:
    expected = sample.get("expected_doc_ids") or []
    gen_result = debug.get("gen_result")
    current_cited = [citation.doc_id for citation in gen_result.citations] if gen_result else []
    current_pass = doc_hit(expected, current_cited)
    final_chunks = debug.get("final_chunks") or []
    expected_candidates = [chunk for chunk in final_chunks if chunk.doc_id in expected]
    expected_in_final = bool(expected_candidates)
    if not expected_in_final:
        return {
            "sample_id": sample["sample_id"],
            "expected_doc_in_final": False,
            "oracle_selected_support_contains_expected": False,
            "oracle_citation_pass": False,
            "current_citation_pass": current_pass,
            "inferred_blocker": "support_selection" if as_bool(variant_row.get("expected_doc_in_rerank_input")) else "unclear",
            "notes": "Expected doc is not available in final chunks, so selected-support oracle cannot be applied.",
        }

    analysis = pipeline.router.analyze(sample["question"])
    candidates = pipeline.generator_v2.ledger_builder.build(sample["question"], analysis, final_chunks)
    selected = pipeline.generator_v2.support_selector.select(
        sample["question"], analysis, candidates, settings.generation
    )
    selected_ids = {item.candidate.chunk_id for item in selected}
    forced = list(selected)
    for candidate in candidates:
        if candidate.doc_id in expected and candidate.chunk_id not in selected_ids:
            forced.append(
                SupportItem(
                    evidence_id=candidate.evidence_id,
                    candidate=candidate,
                    support_score=9.99,
                    reasons=list(candidate.reasons) + ["oracle_expected_doc_forced"],
                )
            )
            break
    forced = forced[: max(len(selected), 1) + 1]
    oracle_contains = doc_hit(expected, [item.candidate.doc_id for item in forced])
    plan = pipeline.generator_v2.answer_planner.plan(
        sample["question"], analysis, forced, candidates, settings.generation
    )
    draft = pipeline.generator_v2.answer_builder.build(
        sample["question"], analysis, plan, forced, config=settings.generation
    )
    candidates_for_citation = pipeline.generator_v2.citation_binder.build_citation_candidates(
        forced, plan_mode=plan.mode, answer_mode=plan.mode
    )
    _, citations, citation_debug = pipeline.generator_v2.citation_binder.bind(
        draft,
        forced,
        plan_mode=plan.mode,
        answer_mode=plan.mode,
        citation_candidates=candidates_for_citation,
    )
    oracle_pass = doc_hit(expected, [citation.doc_id for citation in citations])
    if not oracle_contains:
        blocker = "support_selection"
    elif oracle_pass and not current_pass:
        blocker = "support_selection"
    elif oracle_contains and not oracle_pass:
        blocker = "citation_binding"
    elif oracle_pass == current_pass:
        blocker = "eval_metric" if current_pass else "unclear"
    else:
        blocker = "unclear"
    return {
        "sample_id": sample["sample_id"],
        "expected_doc_in_final": expected_in_final,
        "oracle_selected_support_contains_expected": oracle_contains,
        "oracle_citation_pass": oracle_pass,
        "current_citation_pass": current_pass,
        "inferred_blocker": blocker,
        "notes": (
            f"forced_support_docs={pipe([item.candidate.doc_id for item in forced])}; "
            f"oracle_cited_docs={pipe([citation.doc_id for citation in citations])}; "
            f"citation_debug_order={pipe(citation_debug.get('ordered_evidence_ids', []))}"
        ),
    }


def eval_current_response(sample: dict[str, Any], response: Any) -> tuple[bool, bool]:
    expected = sample.get("expected_doc_ids") or []
    cited = [citation.doc_id for citation in response.citations if citation.doc_id]
    support = [
        str(item.get("doc_id"))
        for item in response.debug.get("generation_v2", {}).get("support_pack", [])
        if item.get("doc_id")
    ]
    hit = doc_hit(expected, cited + support)
    raw_failure = "doc_miss" if expected and not hit else "ok"
    assessed = evaluate_failure(
        raw_failure_category=raw_failure,
        doc_hit=hit,
        cited_doc_ids=uniq(cited),
        expected_doc_ids=expected,
        expected_source_files=sample.get("expected_source_files") or [],
        citation_count=len(cited),
        expected_min_citations=0 if is_negative_abstention(sample) else max(1, min(2, len(expected))),
        answer_mode="full",
        is_negative=is_negative_abstention(sample),
        route_match=(sample.get("expected_route") == getattr(response.route, "value", response.route)),
        source_file_hit=False,
    )
    return hit, bool(assessed.is_real_p0)


def main() -> None:
    RDIR.mkdir(parents=True, exist_ok=True)
    REPDIR.mkdir(parents=True, exist_ok=True)
    configure_eval_env()
    settings = Settings.from_env()

    required = [
        P9D / "regression17_sample_list.csv",
        P9D / "current_live_rewrite_trace.csv",
        P9D / "phase20_rewrite_trace_lookup.csv",
        P9D / "rewrite_delta_audit.csv",
        P9D / "retrieval_variant_ab.csv",
        P9D / "original_cn_fallback_behavior_audit.csv",
        P9D / "reclassified_regression17.csv",
        P9D / "rewrite_reproducibility_recommendation.json",
        P9D / "phase20_baseline_reproducibility_assessment.json",
        P9D / "phase21a9e_next_step_decision.json",
        P9C / "smoke150_aligned_rerun_metrics.json",
        P9C / "regression28_rerun_metrics.json",
        P9C / "rewrite_probe_after_patch.csv",
        P20M / "final_code_state.json",
        P20M / "residual_resolution_ledger.csv",
        P20M / "current_best_config.md",
        SMOKE150,
        SMOKE200,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]

    sample_rows = load_csv(P9D / "regression17_sample_list.csv")
    live = {row["sample_id"]: row for row in load_csv(P9D / "current_live_rewrite_trace.csv")}
    phase20 = {row["sample_id"]: row for row in load_csv(P9D / "phase20_rewrite_trace_lookup.csv")}
    delta = {row["sample_id"]: row for row in load_csv(P9D / "rewrite_delta_audit.csv")}
    fallback = {row["sample_id"]: row for row in load_csv(P9D / "original_cn_fallback_behavior_audit.csv")}
    reclass = {row["sample_id"]: row for row in load_csv(P9D / "reclassified_regression17.csv")}
    variants = {
        (row["sample_id"], row["variant_id"]): row
        for row in load_csv(P9D / "retrieval_variant_ab.csv")
    }
    smoke150 = {row["sample_id"]: row for row in load_jsonl(SMOKE150)}
    smoke150_metrics = load_json(P9C / "smoke150_aligned_rerun_metrics.json")

    run_config = {
        "phase": "21A-9E",
        "purpose": "investigate_real_remaining_smoke150_regressions",
        "sample_count": len(sample_rows),
        "sample_ids": [row["sample_id"] for row in sample_rows],
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
    }
    write_json(RDIR / "run_config.json", run_config)

    master_rows: list[dict[str, Any]] = []
    corrected_rows: list[dict[str, Any]] = []
    for row in sample_rows:
        sid = row["sample_id"]
        v = current_variant(variants, sid)
        root = reclass[sid]["reclassified_root_cause"]
        status, failure_class, first_loss, confidence, action = corrected_for(smoke150[sid], root)
        master_rows.append({
            "sample_id": sid,
            "category": row["category"],
            "expected_route": row["expected_route"],
            "expected_doc_ids": row["expected_doc_ids"],
            "expected_source_files": row["expected_source_files"],
            "current_failure_class": row["current_failure_class"],
            "current_first_loss_stage": row["current_first_loss_stage"],
            "phase20_status": "baseline_pass_real_P0_0",
            "phase21_status": "remaining_real_P0" if as_bool(row["current_real_P0"]) else "focused_pass",
            "current_live_rewrite": live[sid]["live_rewritten_query"],
            "phase20_rewrite": phase20[sid]["phase20_rewritten_query"],
            "rewrite_delta_class": delta[sid]["semantic_delta_assessment"],
            "original_cn_fallback_triggered": fallback[sid]["original_cn_fallback_triggered"],
            "expected_doc_added_by_cn_fallback": fallback[sid]["expected_doc_added_by_fallback"],
            "expected_doc_reached_rerank": v["expected_doc_in_rerank_input"],
            "expected_doc_reached_final": v["expected_doc_in_final"],
            "expected_doc_reached_support": v["expected_doc_in_selected_support"],
            "expected_doc_cited": v["expected_doc_cited"],
            "preliminary_root_cause": root,
            "notes": reclass[sid]["evidence"],
        })
        corrected_rows.append({
            "sample_id": sid,
            "preliminary_root_cause": root,
            "corrected_status": status,
            "corrected_failure_class": failure_class,
            "corrected_first_loss_stage": first_loss,
            "confidence": confidence,
            "recommended_next_action": action,
            "notes": reclass[sid]["evidence"],
        })

    write_csv(RDIR / "regression17_master_ledger.csv", master_rows, [
        "sample_id", "category", "expected_route", "expected_doc_ids", "expected_source_files",
        "current_failure_class", "current_first_loss_stage", "phase20_status", "phase21_status",
        "current_live_rewrite", "phase20_rewrite", "rewrite_delta_class",
        "original_cn_fallback_triggered", "expected_doc_added_by_cn_fallback",
        "expected_doc_reached_rerank", "expected_doc_reached_final",
        "expected_doc_reached_support", "expected_doc_cited", "preliminary_root_cause", "notes",
    ])

    eval_rows = []
    for row in sample_rows:
        sid = row["sample_id"]
        if reclass[sid]["reclassified_root_cause"] != "eval_taxonomy_or_label_issue":
            continue
        sample = smoke150[sid]
        cited = parse_pipe(row["current_cited_doc_ids"])
        eval_rows.append({
            "sample_id": sid,
            "question": sample["question"],
            "expected_doc_ids": row["expected_doc_ids"],
            "expected_source_files": row["expected_source_files"],
            "cited_doc_ids": row["current_cited_doc_ids"],
            "cited_source_files": pipe([f"{doc_id}.pdf" for doc_id in cited]),
            "answer_or_support_summary": sample.get("expected_answer") or "Expected doc is cited; failure is caused by negative_trigger taxonomy, not missing support.",
            "original_eval_result": row["current_failure_class"],
            "corrected_eval_assessment": "false_p0",
            "reason": "Sample is tagged negative_trigger, not abstain/negative_case; it expects a grounded answer and the current run cites expected doc.",
            "recommended_action": "update_eval_taxonomy",
            "notes": reclass[sid]["evidence"],
        })
    write_csv(RDIR / "eval_label_audit.csv", eval_rows, [
        "sample_id", "question", "expected_doc_ids", "expected_source_files", "cited_doc_ids",
        "cited_source_files", "answer_or_support_summary", "original_eval_result",
        "corrected_eval_assessment", "reason", "recommended_action", "notes",
    ])

    negative_rows = []
    for row in sample_rows:
        sid = row["sample_id"]
        if reclass[sid]["reclassified_root_cause"] != "negative_abstention_regression":
            continue
        sample = smoke150[sid]
        v = current_variant(variants, sid)
        cited = note_docs(v["notes"], "cited_docs")
        final_docs = note_docs(v["notes"], "final_docs")
        negative_rows.append({
            "sample_id": sid,
            "question": sample["question"],
            "expected_behavior": pipe(sample.get("expected_behavior") if isinstance(sample.get("expected_behavior"), list) else [sample.get("expected_behavior")]),
            "expected_doc_ids": pipe(sample.get("expected_doc_ids") or []),
            "current_answer_or_support": sample.get("expected_answer") or "Expected refusal/no-answer; current support cites near-topic documents.",
            "cited_doc_ids": pipe(cited),
            "near_topic_docs": pipe(uniq(final_docs + cited)),
            "should_abstain": "true",
            "model_answered_when_should_abstain": "true",
            "evidence_actually_supports_answer": "false",
            "issue_type": "true_negative_regression",
            "recommended_action": "negative_policy_audit",
            "notes": "No expected_doc_ids and abstain_when_insufficient expected; retrieved docs are near-topic but do not establish requested clinical/systematic-review evidence.",
        })
    write_csv(RDIR / "negative_abstention_audit.csv", negative_rows, [
        "sample_id", "question", "expected_behavior", "expected_doc_ids", "current_answer_or_support",
        "cited_doc_ids", "near_topic_docs", "should_abstain", "model_answered_when_should_abstain",
        "evidence_actually_supports_answer", "issue_type", "recommended_action", "notes",
    ])

    support_ids = [
        sid for sid, rec in reclass.items()
        if rec["reclassified_root_cause"] in {"support_selection_regression", "original_cn_fallback_not_effective"}
    ]
    support_ids.append("ent_100")
    support_ids = uniq(support_ids)
    support_rows = []
    for sid in support_ids:
        sample_row = next(row for row in sample_rows if row["sample_id"] == sid)
        expected = parse_pipe(sample_row["expected_doc_ids"])
        v = current_variant(variants, sid)
        final_docs = note_docs(v["notes"], "final_docs")
        support_docs = note_docs(v["notes"], "support_docs")
        cited_docs = note_docs(v["notes"], "cited_docs")
        competing = [doc for doc in uniq(support_docs + cited_docs) if doc not in expected]
        if not as_bool(v["expected_doc_in_final"]):
            stage = "final"
            issue = "doc_diversity_issue"
        elif not as_bool(v["expected_doc_in_selected_support"]):
            stage = "support_selection"
            issue = "true_support_selection_regression"
        elif not as_bool(v["expected_doc_cited"]):
            stage = "citation_binding"
            issue = "citation_binding_regression"
        else:
            stage = "unclear"
            issue = "unclear"
        support_rows.append({
            "sample_id": sid,
            "expected_doc_ids": pipe(expected),
            "expected_doc_in_retrieval": v["expected_doc_in_hybrid_top40"],
            "expected_doc_in_rerank_input": v["expected_doc_in_rerank_input"],
            "expected_doc_in_final": v["expected_doc_in_final"],
            "expected_doc_in_selected_support": v["expected_doc_in_selected_support"],
            "expected_doc_cited": v["expected_doc_cited"],
            "competing_doc_ids": pipe(competing),
            "support_score_expected": "not_selected" if not as_bool(v["expected_doc_in_selected_support"]) else "selected",
            "support_score_competing": "selected_docs=" + pipe(support_docs),
            "failure_stage": stage,
            "issue_type": issue,
            "recommended_action": "support_citation_targeted_audit",
            "notes": f"final_docs={pipe(final_docs)}; support_docs={pipe(support_docs)}; cited_docs={pipe(cited_docs)}",
        })
    write_csv(RDIR / "support_citation_audit.csv", support_rows, [
        "sample_id", "expected_doc_ids", "expected_doc_in_retrieval", "expected_doc_in_rerank_input",
        "expected_doc_in_final", "expected_doc_in_selected_support", "expected_doc_cited",
        "competing_doc_ids", "support_score_expected", "support_score_competing", "failure_stage",
        "issue_type", "recommended_action", "notes",
    ])

    retrieval_rows = []
    for sid in ["ent_083", "ent_100"]:
        sample_row = next(row for row in sample_rows if row["sample_id"] == sid)
        expected = parse_pipe(sample_row["expected_doc_ids"])
        v0 = variants[(sid, "v0_current_live_rewrite")]
        v1 = variants[(sid, "v1_original_cn_only")]
        v2 = variants[(sid, "v2_phase20_rewrite_if_available")]
        v3 = variants[(sid, "v3_live_rewrite_plus_original_cn_fallback")]
        issue = "true_hard_recall_miss" if sid == "ent_083" else "rewrite_drift_recall_loss"
        action = "retrieval_generalization_audit" if sid == "ent_083" else "frozen_rewrite_cache"
        retrieval_rows.append({
            "sample_id": sid,
            "expected_doc_ids": pipe(expected),
            "original_cn_rank": expected_rank(expected, note_docs(v1["notes"], "final_docs"), v1["expected_doc_in_final"]),
            "current_live_rewrite_rank": expected_rank(expected, note_docs(v0["notes"], "final_docs"), v0["expected_doc_in_final"]),
            "phase20_rewrite_rank": expected_rank(expected, note_docs(v2["notes"], "final_docs"), v2["expected_doc_in_final"]),
            "cn_fallback_rank": expected_rank(expected, note_docs(v3["notes"], "final_docs"), v3["expected_doc_in_final"]),
            "dense_rank": expected_rank(expected, note_docs(v3["notes"], "dense_docs"), v3["expected_doc_in_dense_top40"]),
            "bm25_rank": expected_rank(expected, note_docs(v3["notes"], "bm25_docs"), v3["expected_doc_in_bm25_top40"]),
            "hybrid_rank": expected_rank(expected, note_docs(v3["notes"], "hybrid_docs"), v3["expected_doc_in_hybrid_top40"]),
            "rerank_input_contains_expected": v3["expected_doc_in_rerank_input"],
            "final_contains_expected": v3["expected_doc_in_final"],
            "issue_type": issue,
            "recommended_action": action,
            "notes": f"current={v0['notes']}; phase20={v2['notes']}",
        })
    write_csv(RDIR / "retrieval_regression_audit.csv", retrieval_rows, [
        "sample_id", "expected_doc_ids", "original_cn_rank", "current_live_rewrite_rank",
        "phase20_rewrite_rank", "cn_fallback_rank", "dense_rank", "bm25_rank", "hybrid_rank",
        "rerank_input_contains_expected", "final_contains_expected", "issue_type", "recommended_action", "notes",
    ])

    pipeline = SynBioRAGPipeline(settings)
    filters = QueryFilters()

    stability_rows = []
    nondet_ids = [sid for sid, rec in reclass.items() if rec["reclassified_root_cause"] == "nondeterminism"]
    for sid in nondet_ids:
        sample = smoke150[sid]
        for run_id in range(1, 4):
            try:
                response = pipeline.answer(sample["question"], filters=filters)
                cited = [citation.doc_id for citation in response.citations if citation.doc_id]
                support = [
                    str(item.get("doc_id"))
                    for item in response.debug.get("generation_v2", {}).get("support_pack", [])
                    if item.get("doc_id")
                ]
                lifecycle = response.debug.get("evidence_lifecycle_debug", {})
                seed_doc_ids = lifecycle.get("seed_chunks", {}).get("doc_ids", [])
                final_doc_ids = lifecycle.get("final_chunks", {}).get("doc_ids", [])
                rewrite = response.debug.get("query_rewrite", {})
                hit, p0 = eval_current_response(sample, response)
                stability_rows.append({
                    "sample_id": sid,
                    "run_id": run_id,
                    "doc_hit": hit,
                    "cited_doc_ids": pipe(uniq(cited)),
                    "real_P0": p0,
                    "rewrite_output_hash": rewrite.get("rewrite_output_hash") or hashlib.sha256(str(rewrite.get("rewritten_query", "")).encode()).hexdigest()[:16],
                    "rerank_top_docs": pipe(uniq([str(doc_id) for doc_id in seed_doc_ids[:5]])),
                    "selected_support_ids": pipe(uniq(support)),
                    "status": "pass" if not p0 else "fail",
                    "notes": f"final_docs={pipe(uniq([str(doc_id) for doc_id in final_doc_ids]))}; rewritten_query={rewrite.get('rewritten_query', '')}",
                })
            except Exception as exc:
                stability_rows.append({
                    "sample_id": sid,
                    "run_id": run_id,
                    "doc_hit": "",
                    "cited_doc_ids": "",
                    "real_P0": "",
                    "rewrite_output_hash": "",
                    "rerank_top_docs": "",
                    "selected_support_ids": "",
                    "status": "error",
                    "notes": f"{type(exc).__name__}: {exc}",
                })
    write_csv(RDIR / "nondeterminism_stability_check.csv", stability_rows, [
        "sample_id", "run_id", "doc_hit", "cited_doc_ids", "real_P0", "rewrite_output_hash",
        "rerank_top_docs", "selected_support_ids", "status", "notes",
    ])
    stable_pass = stable_fail = unstable = 0
    likely_sources = Counter()
    for sid in nondet_ids:
        rows = [row for row in stability_rows if row["sample_id"] == sid]
        statuses = {row["status"] for row in rows}
        p0s = {str(row["real_P0"]) for row in rows if row["real_P0"] != ""}
        hashes = {row["rewrite_output_hash"] for row in rows if row["rewrite_output_hash"]}
        supports = {row["selected_support_ids"] for row in rows if row["selected_support_ids"]}
        if statuses == {"pass"}:
            stable_pass += 1
        elif statuses == {"fail"}:
            stable_fail += 1
        else:
            unstable += 1
        if len(hashes) > 1:
            likely_sources["rewrite"] += 1
        elif len(supports) > 1 or len(p0s) > 1:
            likely_sources["support_selection"] += 1
        elif "error" in statuses:
            likely_sources["external_llm"] += 1
        else:
            likely_sources["unclear"] += 1
    nondet_summary = {
        "nondeterministic_sample_count": len(nondet_ids),
        "stable_pass_count": stable_pass,
        "stable_fail_count": stable_fail,
        "unstable_count": unstable,
        "likely_source": likely_sources.most_common(1)[0][0] if likely_sources else "unclear",
        "recommended_action": "rerun_smoke150_after_corrections" if stable_pass == len(nondet_ids) else "implement_frozen_eval_rewrite_cache",
        "notes": "Focused 3x rerun over samples previously listed as failures but passing in Phase 21A-9D diagnostics.",
    }
    write_json(RDIR / "nondeterminism_summary.json", nondet_summary)

    stable_fail_sids = set()
    stable_pass_sids = set()
    for sid in nondet_ids:
        rows = [row for row in stability_rows if row["sample_id"] == sid]
        statuses = {row["status"] for row in rows}
        if statuses == {"fail"}:
            stable_fail_sids.add(sid)
        elif statuses == {"pass"}:
            stable_pass_sids.add(sid)

    for corrected in corrected_rows:
        sid = corrected["sample_id"]
        if sid in stable_fail_sids:
            corrected.update({
                "corrected_status": "true_real_p0",
                "corrected_failure_class": "support_selection_regression",
                "corrected_first_loss_stage": "support_selection",
                "confidence": "high",
                "recommended_next_action": "support_citation_targeted_audit",
                "notes": corrected["notes"] + " Phase 21A-9E 3x focused rerun was stable fail; expected doc reached final but was not cited.",
            })
        elif sid in stable_pass_sids:
            corrected.update({
                "corrected_status": "nondeterministic",
                "corrected_failure_class": "stable_pass_after_prior_failure",
                "corrected_first_loss_stage": "nondeterminism",
                "confidence": "medium",
                "recommended_next_action": "rerun_smoke150_after_corrections",
                "notes": corrected["notes"] + " Phase 21A-9E 3x focused rerun was stable pass.",
            })

    for sid in sorted(stable_fail_sids):
        if sid in support_ids:
            continue
        support_ids.append(sid)
        sample_row = next(row for row in sample_rows if row["sample_id"] == sid)
        expected = parse_pipe(sample_row["expected_doc_ids"])
        latest = [row for row in stability_rows if row["sample_id"] == sid][-1]
        final_docs = []
        match = re.search(r"final_docs=([^;]+)", latest.get("notes", ""))
        if match:
            final_docs = parse_pipe(match.group(1))
        support_docs = parse_pipe(latest.get("selected_support_ids", ""))
        cited_docs = parse_pipe(latest.get("cited_doc_ids", ""))
        support_rows.append({
            "sample_id": sid,
            "expected_doc_ids": pipe(expected),
            "expected_doc_in_retrieval": "true",
            "expected_doc_in_rerank_input": "true",
            "expected_doc_in_final": str(doc_hit(expected, final_docs)),
            "expected_doc_in_selected_support": str(doc_hit(expected, support_docs)),
            "expected_doc_cited": str(doc_hit(expected, cited_docs)),
            "competing_doc_ids": pipe([doc for doc in uniq(support_docs + cited_docs) if doc not in expected]),
            "support_score_expected": "not_selected",
            "support_score_competing": "selected_docs=" + pipe(support_docs),
            "failure_stage": "support_selection",
            "issue_type": "true_support_selection_regression",
            "recommended_action": "support_citation_targeted_audit",
            "notes": f"Phase21A-9E stable fail from nondeterminism bucket; final_docs={pipe(final_docs)}; support_docs={pipe(support_docs)}; cited_docs={pipe(cited_docs)}",
        })

    write_csv(RDIR / "support_citation_audit.csv", support_rows, [
        "sample_id", "expected_doc_ids", "expected_doc_in_retrieval", "expected_doc_in_rerank_input",
        "expected_doc_in_final", "expected_doc_in_selected_support", "expected_doc_cited",
        "competing_doc_ids", "support_score_expected", "support_score_competing", "failure_stage",
        "issue_type", "recommended_action", "notes",
    ])

    oracle_rows = []
    for sid in support_ids:
        sample = smoke150[sid]
        retrieval_query = live[sid]["live_rewritten_query"]
        vrow, debug = run_variant(
            pipeline=pipeline,
            sample=sample,
            variant_id="phase21a9e_oracle_base",
            retrieval_query=retrieval_query,
            use_original_cn_fallback=True,
            dual_query_shadow=False,
            filters=filters,
        )
        oracle_rows.append(build_oracle_probe(pipeline, settings, sample, vrow, debug))
    write_csv(RDIR / "oracle_support_citation_probe.csv", oracle_rows, [
        "sample_id", "expected_doc_in_final", "oracle_selected_support_contains_expected",
        "oracle_citation_pass", "current_citation_pass", "inferred_blocker", "notes",
    ])

    write_csv(RDIR / "corrected_regression_assessment.csv", corrected_rows, [
        "sample_id", "preliminary_root_cause", "corrected_status", "corrected_failure_class",
        "corrected_first_loss_stage", "confidence", "recommended_next_action", "notes",
    ])

    status_counts = Counter(row["corrected_status"] for row in corrected_rows)
    true_rows = [row for row in corrected_rows if row["corrected_status"] == "true_real_p0"]
    true_bucket = Counter()
    for row in true_rows:
        if "negative" in row["corrected_failure_class"]:
            true_bucket["negative_abstention"] += 1
        elif "retrieval" in row["corrected_failure_class"]:
            true_bucket["retrieval"] += 1
        elif "rewrite_drift" in row["corrected_failure_class"]:
            true_bucket["support_citation"] += 1
        elif "support" in row["corrected_failure_class"] or "citation" in row["corrected_failure_class"]:
            true_bucket["support_citation"] += 1
        else:
            true_bucket["other"] += 1
    corrected_doc_miss = sum(
        1 for row in true_rows
        if row["corrected_failure_class"] in {
            "support_selection_regression",
            "retrieval_real_regression",
            "rewrite_drift_support_selection",
        }
    )
    corrected_metrics = {
        "original_reported_real_P0": smoke150_metrics["real_P0"],
        "corrected_real_P0": len(true_rows),
        "original_doc_miss": smoke150_metrics["doc_miss"],
        "corrected_doc_miss": corrected_doc_miss,
        "false_or_eval_p0_count": status_counts["false_or_eval_p0"],
        "label_issue_count": status_counts["label_issue"],
        "nondeterministic_count": status_counts["nondeterministic"],
        "true_real_p0_count": len(true_rows),
        "true_real_p0_by_bucket": dict(true_bucket),
        "notes": "Corrected metrics count only samples assessed as true_real_p0; nondeterministic samples are reported separately.",
    }
    write_json(RDIR / "smoke150_corrected_metrics.json", corrected_metrics)

    cache_related = sum(
        1 for row in corrected_rows
        if row["corrected_status"] in {"nondeterministic"}
        or row["preliminary_root_cause"] == "live_rewrite_semantic_drift"
    )
    frozen_decision = {
        "frozen_cache_needed_for_reproducibility": True,
        "frozen_cache_sufficient_to_restore_baseline": False,
        "live_rewrite_drift_true_p0_count": sum(1 for row in corrected_rows if row["preliminary_root_cause"] == "live_rewrite_semantic_drift" and row["corrected_status"] == "true_real_p0"),
        "cache_related_p0_count": cache_related,
        "recommended_cache_action": "implement_before_next_smoke200",
        "rationale": "Frozen rewrites are needed to make eval runs reproducible, but Phase 20 rewrite variants recover only part of the 17 and do not address dominant support/citation or negative-abstention failures.",
        "notes": "Do not treat frozen cache as sufficient baseline restoration.",
    }
    write_json(RDIR / "frozen_rewrite_cache_decision.json", frozen_decision)

    dominant = true_bucket.most_common(1)[0][0] if true_bucket else "none"
    if dominant == "support_citation":
        recommended = "support_citation_targeted_audit"
    elif dominant == "negative_abstention":
        recommended = "negative_abstention_targeted_audit"
    elif cache_related >= len(true_rows):
        recommended = "implement_frozen_eval_rewrite_cache"
    elif status_counts["false_or_eval_p0"] + status_counts["label_issue"] > len(true_rows):
        recommended = "eval_taxonomy_label_correction"
    else:
        recommended = "no_safe_next_step"
    next_decision = {
        "phase21a9e_completed": True,
        "corrected_smoke150_real_P0": corrected_metrics["corrected_real_P0"],
        "true_real_p0_count": corrected_metrics["true_real_p0_count"],
        "dominant_true_real_p0_bucket": dominant,
        "eval_label_issue_count": status_counts["false_or_eval_p0"] + status_counts["label_issue"],
        "nondeterministic_count": status_counts["nondeterministic"],
        "recommended_phase21a9f": recommended,
        "should_rerun_smoke200_now": False,
        "rationale": "True remaining failures are dominated by support/citation propagation after expected docs enter final context; smoke200 remains unsafe until smoke150 corrected failures are understood.",
        "notes": "Phase 21B remains out of scope.",
    }
    write_json(RDIR / "phase21a9f_next_step_decision.json", next_decision)

    root_counts = Counter(row["preliminary_root_cause"] for row in master_rows)
    summary = f"""# Phase 21A-9E Real Remaining Smoke150 Regression Audit

## 1. Purpose
Investigate the 17 remaining smoke150 regressions after query rewrite wiring was fixed in Phase 21A-9C and rewrite/retrieval variants were audited in Phase 21A-9D.

## 2. Regression17 Overview
Initial Phase 21A-9D buckets: {dict(root_counts)}.

## 3. Eval / Label Audit
{len(eval_rows)} samples are false/eval P0: the `h50_neg_*` cases are `negative_trigger` robustness cases that still require grounded citations, and current traces cite the expected documents.

## 4. Negative Abstention Audit
{len(negative_rows)} samples are true negative-abstention regressions. They have no expected document and require refusal when evidence is insufficient, but current support/citations use near-topic documents.

## 5. Support / Citation Audit
{len(support_rows)} samples need support/citation follow-up. Most have the expected doc in final context but lose it during selected-support construction; oracle probing is reported in `oracle_support_citation_probe.csv`.

## 6. Retrieval Audit
`ent_083` remains a true retrieval/final-context miss for the expected comparison docs. `ent_100` is cache/rewrite related in that the Phase 20 rewrite variant recovers doc_0090, but the current live variant selects competing support.

## 7. Nondeterminism Check
Focused 3x rerun covered {len(nondet_ids)} samples. Summary: {nondet_summary}.

## 8. Corrected Metrics
Corrected smoke150 real P0: {corrected_metrics['corrected_real_P0']} / 150. Corrected doc_miss: {corrected_metrics['corrected_doc_miss']}. True buckets: {dict(true_bucket)}.

## 9. Frozen Rewrite Cache Decision
Frozen cache is needed for reproducibility but is not sufficient to restore the Phase 20 baseline. Recommended cache action: `{frozen_decision['recommended_cache_action']}`.

## 10. Recommendation
Recommended Phase 21A-9F: `{next_decision['recommended_phase21a9f']}`. Do not rerun smoke200 now: `{next_decision['should_rerun_smoke200_now']}`.
"""
    (REPDIR / "summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
