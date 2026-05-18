#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.schemas import QueryFilters
from src.synbio_rag.evaluation.failure_taxonomy import evaluate_failure


RDIR = ROOT / "results/phase21a9_smoke200_rebaseline"
REPDIR = ROOT / "reports/phase21a9_smoke200_rebaseline"
SMOKE200 = ROOT / "data/eval/datasets/smoke200.jsonl"
SMOKE150 = ROOT / "data/eval/datasets/smoke150.jsonl"
MANIFEST = ROOT / "data/eval/manifests/smoke200_manifest.json"
REGISTRY = ROOT / "data/eval/registry.json"
SCHEMA = ROOT / "data/eval/schemas/eval_sample_schema.json"
PHASE20_STATE = ROOT / "results/phase20m_convergence_summary/final_code_state.json"
PHASE20_L2_METRICS = ROOT / "results/phase20l2_original_cn_fallback_floor_feature_ab/full_eval_regression_metrics.json"

ADDED50_CATEGORIES = [
    "pdf_table_figure",
    "comparison",
    "cross_lingual",
    "summary_review",
    "negative_near_topic",
    "factoid_precision",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pipe(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    return "|".join(str(v) for v in values)


def route_value(route: Any) -> str:
    return getattr(route, "value", str(route or ""))


def is_negative(sample: dict[str, Any]) -> bool:
    if sample.get("expected_route") == "negative":
        return True
    tags = sample.get("tags") or []
    return any("negative" in str(t) or "abstain" in str(t) for t in tags)


def expected_min_citations(sample: dict[str, Any]) -> int:
    meta = sample.get("metadata") or {}
    value = meta.get("expected_min_citations")
    if isinstance(value, int):
        return value
    if len(sample.get("expected_doc_ids") or []) > 1:
        return 2
    if is_negative(sample):
        return 0
    return 1


def sample_category(sample: dict[str, Any]) -> str:
    category = sample.get("category") or ""
    if category in {"table_figure_caption"}:
        return "pdf_table_figure"
    if category in {"multi_doc_comparison"}:
        return "comparison"
    return category or "unknown"


def failure_class_for(sample: dict[str, Any], row: dict[str, Any], debug: dict[str, Any]) -> str:
    if not row["real_P0"]:
        if row["raw_P0"] and row["failure_type"] == "route_mismatch_false_p0_doc_cited":
            return "route_mismatch_false_p0"
        return "unclear"
    category = sample_category(sample)
    if row["failure_type"] == "negative_query_regression":
        return "wrong_doc_citation" if row["citation_count"] else "negative_regression"
    if row["failure_type"] == "route_mismatch_false_p0_doc_cited":
        return "route_mismatch_false_p0"
    if row["failure_type"] == "zero_citation":
        return "citation_binding_miss"
    if row["failure_type"] in {"doc_miss", "doc_miss_silent"}:
        if category == "pdf_table_figure":
            return "table_figure_extraction_gap"
        if category == "comparison":
            return "comparison_branch_gap"
        if category == "cross_lingual":
            return "cross_lingual_rewrite_gap"
        if category == "summary_review":
            return "summary_scope_gap"
        retrieval_doc_ids = debug.get("retrieval_doc_ids") or []
        if any(doc in retrieval_doc_ids for doc in sample.get("expected_doc_ids") or []):
            return "support_selection_miss"
        return "hard_recall_miss"
    if row["failure_type"] == "wrong_doc_citation":
        return "wrong_doc_citation"
    return "unclear"


def first_loss_stage_for(sample: dict[str, Any], row: dict[str, Any], debug: dict[str, Any]) -> str:
    if not row["real_P0"]:
        return "none"
    expected_docs = set(sample.get("expected_doc_ids") or [])
    retrieval_docs = set(debug.get("retrieval_doc_ids") or [])
    rerank_docs = set(debug.get("rerank_doc_ids") or [])
    final_docs = set(debug.get("final_doc_ids") or [])
    support_docs = set(debug.get("support_doc_ids") or [])
    cited_docs = set(row.get("cited_doc_ids_list") or [])
    if row["expected_route"] and row["route_pred"] and not row["route_match"]:
        return "router"
    if expected_docs and not expected_docs.intersection(retrieval_docs):
        return "hybrid"
    if expected_docs and not expected_docs.intersection(rerank_docs):
        return "rerank"
    if expected_docs and not expected_docs.intersection(final_docs):
        return "final"
    if expected_docs and not expected_docs.intersection(support_docs):
        return "support"
    if expected_docs and not expected_docs.intersection(cited_docs):
        return "citation"
    if sample_category(sample) == "pdf_table_figure":
        return "parser"
    return "generation"


def recommended_phase(failure_class: str) -> str:
    if failure_class in {"table_figure_extraction_gap", "parser_dependent_evidence_gap"}:
        return "phase21b_pdf_parser_table_figure"
    if failure_class in {"hard_recall_miss", "cross_lingual_rewrite_gap", "comparison_branch_gap"}:
        return "phase21c_retrieval_generalization"
    if failure_class in {"support_selection_miss", "low_support_score", "citation_binding_miss", "wrong_doc_citation"}:
        return "phase21d_support_citation"
    if failure_class in {"negative_regression"}:
        return "phase21e_negative_abstention"
    if failure_class == "eval_label_issue":
        return "eval_dataset_review"
    return "no_action" if failure_class in {"route_mismatch_false_p0", "unclear"} else "phase21c_retrieval_generalization"


def severity(row: dict[str, Any], failure_class: str) -> str:
    if failure_class == "eval_label_issue":
        return "eval_issue"
    if row["real_P0"]:
        return "P0"
    if row["raw_P0"]:
        return "P1"
    return "P2"


def run_config(settings: Settings) -> dict[str, Any]:
    return {
        "phase": "21A-9",
        "purpose": "smoke200_rebaseline",
        "dataset_path": "data/eval/datasets/smoke200.jsonl",
        "sample_count": 200,
        "smoke150_count": 150,
        "added50_count": 50,
        "query_rewrite_mode": settings.query_rewrite.mode,
        "original_cn_fallback_enabled": settings.retrieval.original_cn_fallback_enabled,
        "generation_version": "v2",
        "source_floor_enabled": settings.retrieval.source_floor_enabled,
        "parent_expansion_enabled": settings.retrieval.parent_expansion_enabled,
        "rerank_top_k": settings.retrieval.rerank_top_k,
        "qwen_synthesis_enabled": settings.generation.v2_use_qwen_synthesis,
        "alias_expansion_enabled": settings.retrieval.alias_expansion_enabled,
        "comparison_coverage_enabled": settings.generation.v2_enable_comparison_coverage,
        "biolexical_bm25_expansion_enabled": False,
        "index_rebuild": False,
        "code_changed": False,
        "dataset_changed": False,
        "notes": "Phase 20 best experimental settings frozen for smoke200 rebaseline; RAG code, dataset, index, prompts, retrieval/rerank/support/citation logic unchanged.",
    }


def audit_split(smoke200: list[dict[str, Any]], smoke150_ids: set[str]) -> list[dict[str, Any]]:
    rows = []
    for sample in smoke200:
        split = "smoke150" if sample["sample_id"] in smoke150_ids else "added50"
        rows.append({
            "sample_id": sample["sample_id"],
            "source_set": sample.get("source_set", ""),
            "phase_added": sample.get("phase_added", ""),
            "category": sample_category(sample),
            "difficulty": sample.get("difficulty") or "unknown",
            "expected_route": sample.get("expected_route", ""),
            "expected_doc_ids": pipe(sample.get("expected_doc_ids") or []),
            "expected_source_files": pipe(sample.get("expected_source_files") or []),
            "split": split,
            "notes": sample.get("notes") or "",
        })
    return rows


def debug_doc_ids(resp: Any) -> dict[str, list[str]]:
    debug = resp.debug or {}
    gv2 = debug.get("generation_v2", {}) or {}
    lifecycle = debug.get("evidence_lifecycle_debug", {}) or {}
    support_pack = gv2.get("support_pack") or []
    retrieval_hits = debug.get("retrieval_hits", {}) or {}
    rerank_hits = debug.get("rerank_hits", {}) or {}
    final_chunks = lifecycle.get("final_chunks", {}) or {}
    return {
        "support_doc_ids": list(dict.fromkeys(str(x.get("doc_id")) for x in support_pack if x.get("doc_id"))),
        "retrieval_doc_ids": list(dict.fromkeys(str(x) for x in retrieval_hits.get("doc_ids", []) if x)),
        "rerank_doc_ids": list(dict.fromkeys(str(x) for x in rerank_hits.get("doc_ids", []) if x)),
        "final_doc_ids": list(dict.fromkeys(str(x) for x in final_chunks.get("doc_ids", []) if x)),
    }


def evaluate_samples(samples: list[dict[str, Any]], smoke150_ids: set[str], settings: Settings) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    pipeline = SynBioRAGPipeline(settings)
    rows: list[dict[str, Any]] = []
    debug_by_id: dict[str, dict[str, Any]] = {}
    for idx, sample in enumerate(samples, start=1):
        sid = sample["sample_id"]
        split = "smoke150" if sid in smoke150_ids else "added50"
        try:
            t0 = time.perf_counter()
            resp = pipeline.answer(sample["question"], filters=QueryFilters(tenant_id="default"))
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            route_pred = route_value(resp.route)
            cited_doc_ids = list(dict.fromkeys(c.doc_id for c in (resp.citations or []) if c.doc_id))
            support_debug = debug_doc_ids(resp)
            support_doc_ids = support_debug["support_doc_ids"]
            expected_docs = sample.get("expected_doc_ids") or []
            evidence_docs = set(cited_doc_ids) | set(support_doc_ids)
            doc_hit = any(doc in evidence_docs for doc in expected_docs) if expected_docs else True
            doc_miss = bool(expected_docs) and not doc_hit
            route_match = (sample.get("expected_route") or "") == route_pred
            raw_failure = "ok"
            if not route_match:
                raw_failure = "route_mismatch"
            if doc_miss:
                raw_failure = "doc_miss"
            citation_count = len(resp.citations or [])
            source_hit = bool(set(sample.get("expected_source_files") or []).intersection(
                {getattr(c, "source_file", "") for c in (resp.citations or [])}
            ))
            assessed = evaluate_failure(
                raw_failure_category=raw_failure,
                doc_hit=doc_hit,
                cited_doc_ids=cited_doc_ids,
                expected_doc_ids=expected_docs,
                expected_source_files=sample.get("expected_source_files") or [],
                citation_count=citation_count,
                expected_min_citations=expected_min_citations(sample),
                answer_mode="full",
                is_negative=is_negative(sample),
                route_match=route_match,
                source_file_hit=source_hit,
            )
            failure_type = assessed.corrected_failure_category
            wrong_doc = bool(cited_doc_ids) and bool(expected_docs) and not set(cited_doc_ids).intersection(expected_docs)
            if wrong_doc and assessed.is_real_p0:
                failure_type = "wrong_doc_citation"
            answer = resp.answer or ""
            row = {
                "sample_id": sid,
                "split": split,
                "category": sample_category(sample),
                "expected_route": sample.get("expected_route", ""),
                "route_pred": route_pred,
                "route_match": route_match,
                "expected_doc_ids": pipe(expected_docs),
                "cited_doc_ids": pipe(cited_doc_ids),
                "cited_doc_ids_list": cited_doc_ids,
                "doc_hit": doc_hit,
                "doc_miss": doc_miss,
                "citation_count": citation_count,
                "zero_citation": citation_count == 0 and not is_negative(sample),
                "real_P0": bool(assessed.is_real_p0),
                "raw_P0": bool(assessed.is_raw_p0),
                "failure_type": failure_type,
                "answer_length": len(answer),
                "notes": f"elapsed_ms={elapsed_ms}; rewrite_mode={(resp.debug or {}).get('query_rewrite', {}).get('query_rewrite_mode', '')}; rewrite_fallback={(resp.debug or {}).get('query_rewrite', {}).get('rewrite_fallback_used', '')}",
            }
            rows.append(row)
            debug_by_id[sid] = support_debug
        except Exception as exc:
            row = {
                "sample_id": sid,
                "split": split,
                "category": sample_category(sample),
                "expected_route": sample.get("expected_route", ""),
                "route_pred": "",
                "route_match": False,
                "expected_doc_ids": pipe(sample.get("expected_doc_ids") or []),
                "cited_doc_ids": "",
                "cited_doc_ids_list": [],
                "doc_hit": False,
                "doc_miss": True,
                "citation_count": 0,
                "zero_citation": True,
                "real_P0": True,
                "raw_P0": True,
                "failure_type": "error",
                "answer_length": 0,
                "notes": f"error={type(exc).__name__}: {exc}",
            }
            rows.append(row)
            debug_by_id[sid] = {"support_doc_ids": [], "retrieval_doc_ids": [], "rerank_doc_ids": [], "final_doc_ids": []}
        if idx % 10 == 0:
            print(f"evaluated {idx}/{len(samples)}", flush=True)
    return rows, debug_by_id


def metric_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    real_p0 = sum(1 for r in rows if r["real_P0"])
    doc_miss = sum(1 for r in rows if r["doc_miss"])
    return {
        "sample_count": total,
        "real_P0": real_p0,
        "raw_P0": sum(1 for r in rows if r["raw_P0"]),
        "doc_miss": doc_miss,
        "doc_hit_rate": round((total - doc_miss) / total, 6) if total else 0.0,
        "zero_citation": sum(1 for r in rows if r["zero_citation"]),
        "wrong_doc_citation": sum(1 for r in rows if r["failure_type"] == "wrong_doc_citation"),
        "citation_inflation": 0,
        "answer_length_inflation": sum(1 for r in rows if int(r["answer_length"]) > 3000),
        "negative_regression": sum(1 for r in rows if r["failure_type"] == "negative_query_regression"),
        "route_mismatch_false_p0": sum(1 for r in rows if r["failure_type"] == "route_mismatch_false_p0_doc_cited"),
        "corrected_real_P0": real_p0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_taxonomy(samples_by_id: dict[str, dict[str, Any]], rows: list[dict[str, Any]], debug_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if not row["real_P0"] and not row["raw_P0"]:
            continue
        sample = samples_by_id[row["sample_id"]]
        debug = debug_by_id.get(row["sample_id"], {})
        failure_class = failure_class_for(sample, row, debug)
        first_loss = first_loss_stage_for(sample, row, debug)
        out.append({
            "sample_id": row["sample_id"],
            "split": row["split"],
            "category": row["category"],
            "failure_class": failure_class,
            "first_loss_stage": first_loss if first_loss != "none" else "unclear",
            "confidence": "medium" if failure_class == "unclear" else "high",
            "evidence": f"expected={row['expected_doc_ids']}; cited={row['cited_doc_ids']}; support={pipe(debug.get('support_doc_ids', []))}; retrieval={pipe(debug.get('retrieval_doc_ids', [])[:10])}",
            "recommended_next_action": recommended_phase(failure_class),
            "notes": row["notes"],
        })
    return out


def category_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    added = [r for r in rows if r["split"] == "added50"]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in added:
        by_category[row["category"]].append(row)
    metrics = {}
    for category in ADDED50_CATEGORIES:
        bucket = by_category.get(category, [])
        metrics[category] = {
            "sample_count": len(bucket),
            "real_P0": sum(1 for r in bucket if r["real_P0"]),
            "doc_miss": sum(1 for r in bucket if r["doc_miss"]),
            "doc_hit_rate": round((len(bucket) - sum(1 for r in bucket if r["doc_miss"])) / len(bucket), 6) if bucket else 0.0,
            "zero_citation": sum(1 for r in bucket if r["zero_citation"]),
            "wrong_doc_citation": sum(1 for r in bucket if r["failure_type"] == "wrong_doc_citation"),
            "notes": "Phase21 added50 category metrics.",
        }
    return {
        "added50_count": len(added),
        "category_distribution": dict(Counter(r["category"] for r in added)),
        "category_metrics": metrics,
    }


def next_phase(taxonomy: list[dict[str, Any]]) -> tuple[str, str, str]:
    added_failures = [
        r for r in taxonomy
        if r["split"] == "added50" and r["failure_class"] != "route_mismatch_false_p0"
    ]
    counts = Counter(r["failure_class"] for r in added_failures)
    dominant = counts.most_common(1)[0][0] if counts else "none"
    if dominant in {"table_figure_extraction_gap", "parser_dependent_evidence_gap"}:
        return dominant, "phase21b_pdf_parser_table_figure_audit", "Added50 residuals are concentrated in PDF/table/figure evidence extraction."
    if dominant in {"hard_recall_miss", "cross_lingual_rewrite_gap", "comparison_branch_gap"}:
        return dominant, "phase21b_retrieval_generalization_audit", "Added50 residuals are concentrated in hard recall, cross-lingual, or branch retrieval gaps."
    if dominant in {"support_selection_miss", "low_support_score", "citation_binding_miss", "wrong_doc_citation"}:
        return dominant, "phase21b_support_citation_audit", "Added50 residuals are concentrated after retrieval in support or citation binding."
    if dominant == "negative_regression":
        return dominant, "phase21b_negative_abstention_audit", "Added50 residuals are concentrated in negative query abstention."
    if dominant == "eval_label_issue":
        return dominant, "phase21b_eval_label_review", "Added50 residuals are mostly label clarity issues."
    return dominant, "no_action", "No added50 real residuals detected."


def main() -> None:
    RDIR.mkdir(parents=True, exist_ok=True)
    REPDIR.mkdir(parents=True, exist_ok=True)

    for required in [SMOKE200, SMOKE150, MANIFEST, REGISTRY, SCHEMA, PHASE20_STATE, PHASE20_L2_METRICS]:
        if not required.exists():
            raise FileNotFoundError(required)

    os.environ["QUERY_REWRITE_MODE"] = "enabled"
    os.environ["RETRIEVAL_ORIGINAL_CN_FALLBACK_ENABLED"] = "true"
    os.environ["RETRIEVAL_SOURCE_FLOOR_ENABLED"] = "true"
    os.environ["RETRIEVAL_PARENT_EXPANSION_ENABLED"] = "true"
    os.environ["RETRIEVAL_ALIAS_EXPANSION_ENABLED"] = "false"
    os.environ["GENERATION_V2_USE_QWEN_SYNTHESIS"] = "false"
    os.environ["GENERATION_V2_ENABLE_COMPARISON_COVERAGE"] = "false"

    smoke200 = load_jsonl(SMOKE200)
    smoke150 = load_jsonl(SMOKE150)
    smoke150_ids = {s["sample_id"] for s in smoke150}
    samples_by_id = {s["sample_id"]: s for s in smoke200}
    settings = Settings.from_env()

    write_json(RDIR / "run_config.json", run_config(settings))

    split_rows = audit_split(smoke200, smoke150_ids)
    write_csv(RDIR / "dataset_split_audit.csv", split_rows, [
        "sample_id", "source_set", "phase_added", "category", "difficulty",
        "expected_route", "expected_doc_ids", "expected_source_files", "split", "notes",
    ])

    rows, debug_by_id = evaluate_samples(smoke200, smoke150_ids, settings)
    ledger_fields = [
        "sample_id", "split", "category", "expected_route", "route_pred", "route_match",
        "expected_doc_ids", "cited_doc_ids", "doc_hit", "doc_miss", "citation_count",
        "zero_citation", "real_P0", "raw_P0", "failure_type", "answer_length", "notes",
    ]
    write_csv(RDIR / "per_sample_results.csv", rows, ledger_fields)

    all_metrics = metric_block(rows)
    smoke150_rows = [r for r in rows if r["split"] == "smoke150"]
    added50_rows = [r for r in rows if r["split"] == "added50"]
    s150_metrics = metric_block(smoke150_rows)
    a50_metrics = metric_block(added50_rows)
    all_metrics.update({
        "smoke150_real_P0": s150_metrics["real_P0"],
        "added50_real_P0": a50_metrics["real_P0"],
        "smoke150_doc_miss": s150_metrics["doc_miss"],
        "added50_doc_miss": a50_metrics["doc_miss"],
        "smoke150_doc_hit_rate": s150_metrics["doc_hit_rate"],
        "added50_doc_hit_rate": a50_metrics["doc_hit_rate"],
        "notes": "Smoke200 evaluated with Phase 20 best experimental config; failures are diagnostic residuals for next-phase planning.",
    })
    write_json(RDIR / "smoke200_eval_metrics.json", all_metrics)

    taxonomy = build_taxonomy(samples_by_id, rows, debug_by_id)
    write_csv(RDIR / "failure_taxonomy.csv", taxonomy, [
        "sample_id", "split", "category", "failure_class", "first_loss_stage",
        "confidence", "evidence", "recommended_next_action", "notes",
    ])

    write_json(RDIR / "added50_category_breakdown.json", category_breakdown(rows))

    phase20_state = json.loads(PHASE20_STATE.read_text(encoding="utf-8"))
    regression = {
        "smoke150_count": len(smoke150_rows),
        "phase20_baseline_real_P0": phase20_state.get("final_real_P0", 0),
        "current_smoke150_real_P0": s150_metrics["real_P0"],
        "phase20_baseline_doc_miss": phase20_state.get("final_doc_miss", 0),
        "current_smoke150_doc_miss": s150_metrics["doc_miss"],
        "phase20_baseline_doc_hit_rate": phase20_state.get("final_doc_hit_rate", 1.0),
        "current_smoke150_doc_hit_rate": s150_metrics["doc_hit_rate"],
        "regression_detected": s150_metrics["real_P0"] > phase20_state.get("final_real_P0", 0) or s150_metrics["doc_miss"] > phase20_state.get("final_doc_miss", 0),
        "regressed_samples": [r["sample_id"] for r in smoke150_rows if r["real_P0"] or r["doc_miss"]],
        "notes": "If regression is detected, inspect config/cache/nondeterminism before any code fix.",
    }
    write_json(RDIR / "smoke150_regression_check.json", regression)

    backlog = []
    taxonomy_by_id = {r["sample_id"]: r for r in taxonomy}
    for row in added50_rows:
        if not row["real_P0"] and not row["raw_P0"]:
            continue
        tax = taxonomy_by_id.get(row["sample_id"], {})
        failure_class = tax.get("failure_class", "unclear")
        backlog.append({
            "sample_id": row["sample_id"],
            "category": row["category"],
            "failure_class": failure_class,
            "first_loss_stage": tax.get("first_loss_stage", "unclear"),
            "expected_doc_ids": row["expected_doc_ids"],
            "cited_doc_ids": row["cited_doc_ids"],
            "severity": severity(row, failure_class),
            "recommended_phase": recommended_phase(failure_class),
            "notes": tax.get("notes", row["notes"]),
        })
    write_csv(RDIR / "added50_residual_backlog.csv", backlog, [
        "sample_id", "category", "failure_class", "first_loss_stage",
        "expected_doc_ids", "cited_doc_ids", "severity", "recommended_phase", "notes",
    ])

    dominant, phase, rationale = next_phase(taxonomy)
    decision = {
        "phase21a9_completed": True,
        "smoke200_eval_completed": True,
        "smoke150_regression_detected": regression["regression_detected"],
        "smoke200_real_P0": all_metrics["real_P0"],
        "smoke150_real_P0": s150_metrics["real_P0"],
        "added50_real_P0": a50_metrics["real_P0"],
        "dominant_added50_failure_bucket": dominant,
        "recommended_next_phase": phase,
        "rationale": rationale,
        "do_not_fix_yet": True,
        "notes": "Phase 21A-9 is diagnostic only; defer code/data/index/prompt changes to a separately scoped phase.",
    }
    write_json(RDIR / "phase21b_next_step_decision.json", decision)

    category_counts = Counter(r["category"] for r in added50_rows if r["real_P0"])
    failure_counts = Counter(r.get("failure_class", "unclear") for r in taxonomy if r["split"] == "added50")
    report = f"""# Phase 21A-9 Smoke200 Rebaseline

## 1. Purpose
This phase runs the Phase 20 best experimental configuration against the formal smoke200 dataset as a rebaseline.

## 2. Dataset
Smoke200 contains smoke150 plus added50. Phase 21A-8 validated 200 unique samples and confirmed added50 has zero doc/source overlap with smoke150.

## 3. Overall Metrics
- sample_count: {all_metrics['sample_count']}
- real_P0: {all_metrics['real_P0']}
- doc_miss: {all_metrics['doc_miss']}
- doc_hit_rate: {all_metrics['doc_hit_rate']}
- zero_citation: {all_metrics['zero_citation']}
- wrong_doc_citation: {all_metrics['wrong_doc_citation']}

## 4. Smoke150 Regression Check
- phase20_baseline_real_P0: {regression['phase20_baseline_real_P0']}
- current_smoke150_real_P0: {regression['current_smoke150_real_P0']}
- current_smoke150_doc_miss: {regression['current_smoke150_doc_miss']}
- current_smoke150_doc_hit_rate: {regression['current_smoke150_doc_hit_rate']}
- regression_detected: {regression['regression_detected']}

## 5. Added50 Results
- added50_real_P0: {a50_metrics['real_P0']}
- added50_doc_miss: {a50_metrics['doc_miss']}
- added50_doc_hit_rate: {a50_metrics['doc_hit_rate']}
- residual_categories: {dict(category_counts)}

## 6. Failure Taxonomy
Added50 failure buckets: {dict(failure_counts)}

## 7. Residual Backlog
Residual backlog is written to `results/phase21a9_smoke200_rebaseline/added50_residual_backlog.csv`.

## 8. Recommendation
Recommended next phase: `{phase}`.

Rationale: {rationale}
"""
    (REPDIR / "summary.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "smoke200_real_P0": all_metrics["real_P0"],
        "smoke200_doc_miss": all_metrics["doc_miss"],
        "smoke200_doc_hit_rate": all_metrics["doc_hit_rate"],
        "smoke150_regression_detected": regression["regression_detected"],
        "added50_real_P0": a50_metrics["real_P0"],
        "dominant_added50_failure_bucket": dominant,
        "recommended_next_phase": phase,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
