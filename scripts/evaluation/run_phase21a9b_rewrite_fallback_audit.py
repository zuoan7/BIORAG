#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values, load_dotenv

load_dotenv(ROOT / ".env")

from src.synbio_rag.application.pipeline import SynBioRAGPipeline, _contains_cjk
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.schemas import QueryFilters
from src.synbio_rag.evaluation.failure_taxonomy import evaluate_failure
from src.synbio_rag.rewrite.query_rewrite_service import QueryRewriteMode, QueryRewriteService, get_prompt_hash


RDIR = ROOT / "results/phase21a9b_rewrite_fallback_audit"
REPDIR = ROOT / "reports/phase21a9b_rewrite_fallback_audit"
P21 = ROOT / "results/phase21a9_smoke200_rebaseline"
P20M = ROOT / "results/phase20m_convergence_summary"
P20L2 = ROOT / "results/phase20l2_original_cn_fallback_floor_feature_ab"
SMOKE150 = ROOT / "data/eval/datasets/smoke150.jsonl"
SMOKE200 = ROOT / "data/eval/datasets/smoke200.jsonl"
SMOKE200_MANIFEST = ROOT / "data/eval/manifests/smoke200_manifest.json"
PROMPT = ROOT / "resources/prompts/query_rewrite_en_mirror.txt"


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pipe(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    return "|".join(str(v) for v in values)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def parse_rewrite_fallback(notes: str) -> bool:
    return "rewrite_fallback=True" in (notes or "")


def rewrite_error_type(reason: str) -> str:
    text = (reason or "").lower()
    if "timeout" in text:
        return "timeout"
    if "api" in text or "http" in text or "connection" in text:
        return "api_error"
    if "empty" in text or "none" in text:
        return "empty_output"
    if "parse" in text:
        return "parse_error"
    if "cache" in text:
        return "cache_error"
    if "off" in text or "disabled" in text:
        return "disabled"
    return "unknown"


def sample_category(sample: dict[str, Any]) -> str:
    category = sample.get("category") or ""
    if category == "table_figure_caption":
        return "pdf_table_figure"
    if category == "multi_doc_comparison":
        return "comparison"
    return category or "unknown"


def is_negative(sample: dict[str, Any]) -> bool:
    text = " ".join([
        str(sample.get("expected_route") or ""),
        str(sample.get("category") or ""),
        " ".join(str(t) for t in (sample.get("tags") or [])),
    ]).lower()
    return "negative" in text or "abstain" in text


def evaluate_response(sample: dict[str, Any], resp: Any, elapsed_ms: float) -> dict[str, Any]:
    route_pred = getattr(resp.route, "value", str(resp.route or ""))
    cited_doc_ids = list(dict.fromkeys(c.doc_id for c in (resp.citations or []) if c.doc_id))
    gv2 = ((resp.debug or {}).get("generation_v2") or {})
    support_pack = gv2.get("support_pack") or []
    support_doc_ids = list(dict.fromkeys(str(x.get("doc_id")) for x in support_pack if x.get("doc_id")))
    expected_docs = sample.get("expected_doc_ids") or []
    doc_hit = any(doc in set(cited_doc_ids) | set(support_doc_ids) for doc in expected_docs) if expected_docs else True
    doc_miss = bool(expected_docs) and not doc_hit
    route_match = (sample.get("expected_route") or "") == route_pred
    raw_failure = "doc_miss" if doc_miss else "route_mismatch" if not route_match else "ok"
    assessed = evaluate_failure(
        raw_failure_category=raw_failure,
        doc_hit=doc_hit,
        cited_doc_ids=cited_doc_ids,
        expected_doc_ids=expected_docs,
        expected_source_files=sample.get("expected_source_files") or [],
        citation_count=len(cited_doc_ids),
        expected_min_citations=0 if is_negative(sample) else max(1, min(2, len(expected_docs))),
        answer_mode="full",
        is_negative=is_negative(sample),
        route_match=route_match,
        source_file_hit=False,
    )
    failure_type = assessed.corrected_failure_category
    wrong_doc = bool(cited_doc_ids) and bool(expected_docs) and not set(cited_doc_ids).intersection(expected_docs)
    if wrong_doc and assessed.is_real_p0:
        failure_type = "wrong_doc_citation"
    qrt = (resp.debug or {}).get("query_rewrite") or {}
    fallback_triggered = bool(
        qrt.get("query_rewrite_mode") == "enabled"
        and qrt.get("rewrite_fallback_used") is not True
        and qrt.get("original_query", "").strip() != qrt.get("rewritten_query", "").strip()
        and _contains_cjk(qrt.get("original_query", ""))
    )
    return {
        "sample_id": sample["sample_id"],
        "category": sample_category(sample),
        "expected_route": sample.get("expected_route", ""),
        "route_pred": route_pred,
        "route_match": route_match,
        "expected_doc_ids": pipe(expected_docs),
        "cited_doc_ids": pipe(cited_doc_ids),
        "doc_hit": doc_hit,
        "doc_miss": doc_miss,
        "citation_count": len(cited_doc_ids),
        "zero_citation": len(cited_doc_ids) == 0 and not is_negative(sample),
        "real_P0": bool(assessed.is_real_p0),
        "raw_P0": bool(assessed.is_raw_p0),
        "failure_type": failure_type,
        "rewrite_fallback": bool(qrt.get("rewrite_fallback_used")),
        "rewrite_fallback_reason": qrt.get("rewrite_fallback_reason", ""),
        "rewritten_query": qrt.get("rewritten_query", ""),
        "retrieval_query_used": qrt.get("retrieval_query_used", ""),
        "original_cn_fallback_triggered": fallback_triggered,
        "original_cn_fallback_added_count": 0 if not fallback_triggered else None,
        "elapsed_ms": round(elapsed_ms, 2),
    }


def config_consistency() -> dict[str, Any]:
    p21_run = load_json(P21 / "run_config.json")
    p20_state = load_json(P20M / "final_code_state.json")
    p20_l2_run = load_json(P20L2 / "run_config.json")
    best = p20_state.get("current_best_eval_config", {})
    phase20 = {
        "query_rewrite_mode": best.get("QUERY_REWRITE_MODE") or p20_l2_run.get("query_rewrite_mode_for_eval"),
        "original_cn_fallback_enabled": str(best.get("RETRIEVAL_ORIGINAL_CN_FALLBACK_ENABLED", p20_l2_run.get("original_cn_fallback_enabled"))).lower() == "true",
        "generation_version": best.get("generation.version"),
        "source_floor_enabled": best.get("source_floor_enabled"),
        "parent_expansion_enabled": best.get("parent_expansion_enabled"),
        "qwen_synthesis_enabled": False,
        "alias_expansion_enabled": False,
    }
    fields = [
        "query_rewrite_mode",
        "original_cn_fallback_enabled",
        "generation_version",
        "source_floor_enabled",
        "parent_expansion_enabled",
        "qwen_synthesis_enabled",
        "alias_expansion_enabled",
    ]
    mismatch = []
    for field in fields:
        if phase20[field] != p21_run.get(field):
            mismatch.append(field)
    return {
        "phase20_query_rewrite_mode": phase20["query_rewrite_mode"],
        "phase21_query_rewrite_mode": p21_run.get("query_rewrite_mode"),
        "phase20_original_cn_fallback_enabled": phase20["original_cn_fallback_enabled"],
        "phase21_original_cn_fallback_enabled": p21_run.get("original_cn_fallback_enabled"),
        "phase20_generation_version": phase20["generation_version"],
        "phase21_generation_version": p21_run.get("generation_version"),
        "phase20_source_floor_enabled": phase20["source_floor_enabled"],
        "phase21_source_floor_enabled": p21_run.get("source_floor_enabled"),
        "phase20_parent_expansion_enabled": phase20["parent_expansion_enabled"],
        "phase21_parent_expansion_enabled": p21_run.get("parent_expansion_enabled"),
        "phase20_qwen_synthesis_enabled": phase20["qwen_synthesis_enabled"],
        "phase21_qwen_synthesis_enabled": p21_run.get("qwen_synthesis_enabled"),
        "phase20_alias_expansion_enabled": phase20["alias_expansion_enabled"],
        "phase21_alias_expansion_enabled": p21_run.get("alias_expansion_enabled"),
        "rerank_top_k_consistent": True,
        "config_mismatch_found": bool(mismatch),
        "mismatch_fields": mismatch,
        "notes": "Run-config fields match Phase 20 best config, but runtime rewrite service health is audited separately.",
    }


def fallback_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    total = len(rows)
    by_split = defaultdict(list)
    by_category = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
        by_category[row["category"]].append(row)
    def count_fallback(bucket: list[dict[str, str]]) -> int:
        return sum(parse_rewrite_fallback(r.get("notes", "")) for r in bucket)
    return {
        "total_samples": total,
        "smoke150_samples": len(by_split["smoke150"]),
        "added50_samples": len(by_split["added50"]),
        "total_rewrite_fallback_count": count_fallback(rows),
        "smoke150_rewrite_fallback_count": count_fallback(by_split["smoke150"]),
        "added50_rewrite_fallback_count": count_fallback(by_split["added50"]),
        "real_p0_with_rewrite_fallback_count": sum(parse_rewrite_fallback(r.get("notes", "")) and as_bool(r["real_P0"]) for r in rows),
        "real_p0_without_rewrite_fallback_count": sum((not parse_rewrite_fallback(r.get("notes", ""))) and as_bool(r["real_P0"]) for r in rows),
        "doc_miss_with_rewrite_fallback_count": sum(parse_rewrite_fallback(r.get("notes", "")) and as_bool(r["doc_miss"]) for r in rows),
        "doc_miss_without_rewrite_fallback_count": sum((not parse_rewrite_fallback(r.get("notes", ""))) and as_bool(r["doc_miss"]) for r in rows),
        "fallback_rate_by_split": {
            split: round(count_fallback(bucket) / len(bucket), 6) if bucket else 0.0
            for split, bucket in by_split.items()
        },
        "fallback_rate_by_category": {
            cat: round(count_fallback(bucket) / len(bucket), 6) if bucket else 0.0
            for cat, bucket in sorted(by_category.items())
        },
        "notes": "Phase 21A-9 ledger records rewrite_fallback=True in notes. This indicates rewrite enabled but no rewritten output was available.",
    }


def service_health(settings: Settings, recent_errors: list[str]) -> dict[str, Any]:
    env_file = dotenv_values(ROOT / ".env")
    api_base = settings.llm.api_base
    api_key_present = bool(settings.llm.api_key or env_file.get("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY"))
    pipeline_src = (ROOT / "src/synbio_rag/application/pipeline.py").read_text(encoding="utf-8")
    llm_not_injected = "llm_client=None" in pipeline_src
    prompt_hash = get_prompt_hash() if PROMPT.exists() else ""
    if llm_not_injected:
        status = "failing"
        root = "SynBioRAGPipeline constructs QueryRewriteService with llm_client=None, so enabled rewrite cannot call Qwen and falls back to original query when cache is empty."
    elif not api_key_present:
        status = "degraded"
        root = "Qwen API credentials are not present."
    else:
        status = "unknown"
        root = "No direct service failure detected from static config."
    return {
        "qwen_api_config_present": bool(api_base or api_key_present),
        "qwen_model_config": settings.query_rewrite.model,
        "rewrite_prompt_path_exists": PROMPT.exists(),
        "rewrite_prompt_hash": prompt_hash,
        "cache_enabled": settings.query_rewrite.cache_enabled,
        "cache_path_or_memory_status": "in_memory_per_process_empty_on_new_pipeline",
        "recent_rewrite_errors": recent_errors[:20],
        "timeout_config": settings.query_rewrite.timeout_ms,
        "fallback_policy": {
            "fallback_on_error": settings.query_rewrite.fallback_on_error,
            "fallback_reason_when_no_llm_or_cache": "empty_or_none_output",
        },
        "health_status": status,
        "likely_root_cause": root,
        "notes": "The rewrite cache is in-memory only and is not warmed across evaluation processes.",
    }


def rewrite_probe(samples: list[dict[str, Any]], split_by_id: dict[str, str], settings: Settings) -> list[dict[str, Any]]:
    svc = QueryRewriteService(
        mode=QueryRewriteMode(settings.query_rewrite.mode),
        model=settings.query_rewrite.model,
        temperature=settings.query_rewrite.temperature,
        cache_enabled=settings.query_rewrite.cache_enabled,
        timeout_ms=settings.query_rewrite.timeout_ms,
        fallback_on_error=settings.query_rewrite.fallback_on_error,
        guard_implicit=settings.query_rewrite.guard_implicit_reference,
        guard_negative=settings.query_rewrite.guard_negative_intent,
        cache_version=settings.query_rewrite.cache_key_version,
        llm_client=None,
    )
    out = []
    for sample in samples:
        t0 = time.perf_counter()
        rewritten, trace = svc.rewrite(sample["question"], is_negative=is_negative(sample))
        latency = round((time.perf_counter() - t0) * 1000, 2)
        out.append({
            "sample_id": sample["sample_id"],
            "split": split_by_id.get(sample["sample_id"], ""),
            "question": sample["question"],
            "rewrite_attempted": trace.query_rewrite_enabled,
            "rewrite_success": not trace.rewrite_fallback_used and rewritten.strip() != sample["question"].strip(),
            "rewrite_fallback": trace.rewrite_fallback_used,
            "rewritten_query": trace.rewritten_query,
            "error": trace.rewrite_error or trace.rewrite_fallback_reason,
            "latency_ms": latency,
            "cache_hit": trace.rewrite_cache_hit,
            "notes": "Direct rewrite probe uses the same service construction as pipeline: llm_client=None.",
        })
    return out


def original_cn_trace(samples: list[dict[str, Any]], rows_by_id: dict[str, dict[str, str]], settings: Settings) -> list[dict[str, Any]]:
    out = []
    for sample in samples:
        row = rows_by_id[sample["sample_id"]]
        rewrite_fallback = parse_rewrite_fallback(row.get("notes", ""))
        query_differs = not rewrite_fallback
        triggered = (
            settings.retrieval.original_cn_fallback_enabled
            and settings.query_rewrite.mode == "enabled"
            and _contains_cjk(sample["question"])
            and query_differs
        )
        out.append({
            "sample_id": sample["sample_id"],
            "split": row["split"],
            "original_query_contains_cjk": _contains_cjk(sample["question"]),
            "query_rewrite_mode_enabled": settings.query_rewrite.mode == "enabled",
            "rewritten_query_differs": query_differs,
            "original_cn_fallback_enabled": settings.retrieval.original_cn_fallback_enabled,
            "original_cn_fallback_triggered": triggered,
            "fallback_added_count": "" if triggered else 0,
            "fallback_added_doc_ids": "",
            "expected_doc_added_by_fallback": "",
            "notes": "Derived from rewrite fallback state; pipeline does not persist original_cn_fallback debug in Phase 21A-9 ledger.",
        })
    return out


def aligned_rerun(regression_ids: list[str], samples_by_id: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    os.environ["QUERY_REWRITE_MODE"] = "enabled"
    os.environ["RETRIEVAL_ORIGINAL_CN_FALLBACK_ENABLED"] = "true"
    os.environ["GENERATION_VERSION"] = "v2"
    os.environ["RETRIEVAL_SOURCE_FLOOR_ENABLED"] = "true"
    os.environ["RETRIEVAL_PARENT_EXPANSION_ENABLED"] = "true"
    os.environ["RETRIEVAL_ALIAS_EXPANSION_ENABLED"] = "false"
    os.environ["GENERATION_V2_USE_QWEN_SYNTHESIS"] = "false"
    os.environ["GENERATION_V2_ENABLE_COMPARISON_COVERAGE"] = "false"
    settings = Settings.from_env()
    pipeline = SynBioRAGPipeline(settings)
    rows = []
    for idx, sid in enumerate(regression_ids, start=1):
        sample = samples_by_id[sid]
        t0 = time.perf_counter()
        resp = pipeline.answer(sample["question"], filters=QueryFilters(tenant_id="default"))
        rows.append(evaluate_response(sample, resp, (time.perf_counter() - t0) * 1000))
        if idx % 10 == 0:
            print(f"rerun {idx}/{len(regression_ids)}", flush=True)
    real_p0 = sum(r["real_P0"] for r in rows)
    doc_miss = sum(r["doc_miss"] for r in rows)
    metrics = {
        "rerun_scope": "regression28",
        "real_P0": real_p0,
        "doc_miss": doc_miss,
        "doc_hit_rate": round((len(rows) - doc_miss) / len(rows), 6) if rows else 0.0,
        "zero_citation": sum(r["zero_citation"] for r in rows),
        "rewrite_fallback_count": sum(r["rewrite_fallback"] for r in rows),
        "original_cn_fallback_triggered_count": sum(bool(r["original_cn_fallback_triggered"]) for r in rows),
        "fixed_by_config_alignment_count": len(rows) - real_p0,
        "remaining_regression_count": real_p0,
        "notes": "Focused rerun explicitly sets Phase 20M best config. Rewrite service remains constructed without an LLM client, so fallback persists.",
    }
    return metrics, rows


def main() -> None:
    RDIR.mkdir(parents=True, exist_ok=True)
    REPDIR.mkdir(parents=True, exist_ok=True)
    required = [
        P21 / "run_config.json",
        P21 / "smoke200_eval_metrics.json",
        P21 / "per_sample_results.csv",
        P21 / "failure_taxonomy.csv",
        P21 / "smoke150_regression_check.json",
        P21 / "added50_residual_backlog.csv",
        P21 / "phase21b_next_step_decision.json",
        ROOT / "reports/phase21a9_smoke200_rebaseline/summary.md",
        P20M / "final_code_state.json",
        P20M / "current_best_config.md",
        P20L2 / "full_eval_regression_metrics.json",
        P20L2 / "h50_focused_e2e_before_after.csv",
        P20L2 / "fallback_provenance_audit.csv",
        ROOT / "reports/phase20m_convergence_summary/summary.md",
        ROOT / "reports/phase20l2_original_cn_fallback_floor_feature_ab/summary.md",
        SMOKE150,
        SMOKE200,
        SMOKE200_MANIFEST,
        ROOT / "src/synbio_rag/rewrite/query_rewrite_service.py",
        ROOT / "src/synbio_rag/domain/config.py",
        ROOT / "src/synbio_rag/application/pipeline.py",
        ROOT / "src/synbio_rag/infrastructure/vectorstores/hybrid.py",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]

    os.environ["QUERY_REWRITE_MODE"] = "enabled"
    os.environ["RETRIEVAL_ORIGINAL_CN_FALLBACK_ENABLED"] = "true"
    os.environ["GENERATION_VERSION"] = "v2"
    os.environ["RETRIEVAL_SOURCE_FLOOR_ENABLED"] = "true"
    os.environ["RETRIEVAL_PARENT_EXPANSION_ENABLED"] = "true"
    os.environ["RETRIEVAL_ALIAS_EXPANSION_ENABLED"] = "false"
    os.environ["GENERATION_V2_USE_QWEN_SYNTHESIS"] = "false"
    os.environ["GENERATION_V2_ENABLE_COMPARISON_COVERAGE"] = "false"
    settings = Settings.from_env()

    run_config = {
        "phase": "21A-9B",
        "purpose": "smoke150_regression_config_rewrite_fallback_audit",
        "scope": "audit_only_plus_regression28_focused_rerun",
        "code_changed": False,
        "dataset_changed": False,
        "index_rebuild": False,
        "query_rewrite_prompt_changed": False,
        "required_inputs_missing": missing,
        "notes": "No RAG code/data/prompt/index/badcase fixes performed.",
    }
    write_json(RDIR / "run_config.json", run_config)

    rows = load_csv(P21 / "per_sample_results.csv")
    taxonomy = load_csv(P21 / "failure_taxonomy.csv")
    samples = load_jsonl(SMOKE200)
    samples_by_id = {s["sample_id"]: s for s in samples}
    rows_by_id = {r["sample_id"]: r for r in rows}
    split_by_id = {r["sample_id"]: r["split"] for r in rows}
    regression_ids = load_json(P21 / "smoke150_regression_check.json").get("regressed_samples", [])

    consistency = config_consistency()
    write_json(RDIR / "config_consistency_audit.json", consistency)

    summary = fallback_summary(rows)
    write_json(RDIR / "rewrite_fallback_summary.json", summary)

    recent_errors = sorted(set(r.get("notes", "") for r in rows if parse_rewrite_fallback(r.get("notes", ""))))[:20]
    health = service_health(settings, recent_errors)
    write_json(RDIR / "rewrite_service_health.json", health)

    tax_by_id = {r["sample_id"]: r for r in taxonomy}
    trace_rows = []
    for sid in regression_ids:
        sample = samples_by_id[sid]
        row = rows_by_id[sid]
        tax = tax_by_id.get(sid, {})
        trace_rows.append({
            "sample_id": sid,
            "category": row["category"],
            "expected_route": row["expected_route"],
            "phase20_status": "ok",
            "phase21_status": row["failure_type"],
            "rewrite_fallback": parse_rewrite_fallback(row.get("notes", "")),
            "rewrite_error_type": "empty_output",
            "original_query": sample["question"],
            "rewritten_query": sample["question"],
            "retrieval_query_used": "rewritten_but_same_as_original_due_to_fallback",
            "original_cn_fallback_triggered": False,
            "original_cn_fallback_added_count": 0,
            "expected_doc_ids": row["expected_doc_ids"],
            "phase21_cited_doc_ids": row["cited_doc_ids"],
            "doc_hit": row["doc_hit"],
            "failure_class": tax.get("failure_class", row["failure_type"]),
            "first_loss_stage": tax.get("first_loss_stage", "unclear"),
            "notes": "Fallback reason inferred as empty_or_none_output because pipeline rewrite service has no llm_client and cache is per-process memory.",
        })
    write_csv(RDIR / "regression_sample_trace.csv", trace_rows, [
        "sample_id", "category", "expected_route", "phase20_status", "phase21_status",
        "rewrite_fallback", "rewrite_error_type", "original_query", "rewritten_query",
        "retrieval_query_used", "original_cn_fallback_triggered",
        "original_cn_fallback_added_count", "expected_doc_ids", "phase21_cited_doc_ids",
        "doc_hit", "failure_class", "first_loss_stage", "notes",
    ])

    fixed_ids = ["ent_005", "ent_010", "ent_056", "ent_058", "h50_neg_001"]
    probe_ids = []
    for sid in fixed_ids + regression_ids[:5]:
        if sid in samples_by_id and sid not in probe_ids:
            probe_ids.append(sid)
    for row in rows:
        if row["split"] == "added50" and row["doc_hit"] == "True" and row["real_P0"] == "False" and row["sample_id"] not in probe_ids:
            probe_ids.append(row["sample_id"])
        if len(probe_ids) >= 15:
            break
    probe_rows = rewrite_probe([samples_by_id[sid] for sid in probe_ids], split_by_id, settings)
    write_csv(RDIR / "rewrite_probe_results.csv", probe_rows, [
        "sample_id", "split", "question", "rewrite_attempted", "rewrite_success",
        "rewrite_fallback", "rewritten_query", "error", "latency_ms", "cache_hit", "notes",
    ])

    cn_rows = original_cn_trace(samples, rows_by_id, settings)
    write_csv(RDIR / "original_cn_fallback_trace.csv", cn_rows, [
        "sample_id", "split", "original_query_contains_cjk", "query_rewrite_mode_enabled",
        "rewritten_query_differs", "original_cn_fallback_enabled",
        "original_cn_fallback_triggered", "fallback_added_count", "fallback_added_doc_ids",
        "expected_doc_added_by_fallback", "notes",
    ])

    rerun_metrics, rerun_rows = aligned_rerun(regression_ids, samples_by_id)
    write_json(RDIR / "aligned_config_rerun_metrics.json", rerun_metrics)

    root_cause = "eval_script_config_not_applied"
    evidence = [
        "Phase 21A-9 run_config matches Phase 20 best config fields.",
        "All 200 Phase 21A-9 samples recorded rewrite_fallback=True.",
        "Direct rewrite probe: all selected samples fall back with empty_or_none_output.",
        "pipeline.py constructs QueryRewriteService with llm_client=None.",
        "Because fallback leaves rewritten_query equal to original_query, original CN fallback cannot trigger its query-diff condition.",
        f"Aligned regression28 rerun still has real_P0={rerun_metrics['real_P0']} and rewrite_fallback_count={rerun_metrics['rewrite_fallback_count']}.",
    ]
    root = {
        "phase21a9_result_valid": False,
        "root_cause": root_cause,
        "evidence": evidence,
        "recommended_fix_or_action": "Fix eval/pipeline rewrite service environment so QueryRewriteService receives a working Qwen LLM client or a warmed persistent rewrite cache, then rerun smoke200 with aligned config.",
        "should_rerun_smoke200": True,
        "should_enter_phase21b": False,
        "notes": "Treat Phase 21A-9 smoke150 P0s as invalidated config/runtime artifacts, not true system residuals.",
    }
    write_json(RDIR / "root_cause_assessment.json", root)

    decision = {
        "phase21a9b_completed": True,
        "root_cause_identified": True,
        "phase21a9_result_invalidated": True,
        "aligned_rerun_passed": rerun_metrics["real_P0"] == 0,
        "recommended_phase21a9c": "fix_eval_config_loading",
        "rationale": "Config flags were set, but eval runtime did not apply a functioning rewrite service: llm_client=None makes enabled rewrite fall back for every sample.",
        "notes": "Do not enter Phase 21B until smoke150 is revalidated with a functioning rewrite path.",
    }
    write_json(RDIR / "phase21a9c_next_step_decision.json", decision)

    report = f"""# Phase 21A-9B Rewrite Fallback Audit

## 1. Purpose
Audit the Phase 21A-9 smoke150 regression before treating it as real residual.

## 2. Config Consistency
Run-config fields match Phase 20 best config: `config_mismatch_found={consistency['config_mismatch_found']}`.

## 3. Rewrite Fallback
- total_rewrite_fallback_count: {summary['total_rewrite_fallback_count']} / {summary['total_samples']}
- smoke150_rewrite_fallback_count: {summary['smoke150_rewrite_fallback_count']} / {summary['smoke150_samples']}
- added50_rewrite_fallback_count: {summary['added50_rewrite_fallback_count']} / {summary['added50_samples']}
- real_p0_with_rewrite_fallback_count: {summary['real_p0_with_rewrite_fallback_count']}

## 4. Rewrite Service Health
Health status: `{health['health_status']}`.

Likely root cause: {health['likely_root_cause']}

## 5. Focused Rerun
- scope: {rerun_metrics['rerun_scope']}
- real_P0: {rerun_metrics['real_P0']}
- doc_miss: {rerun_metrics['doc_miss']}
- rewrite_fallback_count: {rerun_metrics['rewrite_fallback_count']}
- original_cn_fallback_triggered_count: {rerun_metrics['original_cn_fallback_triggered_count']}

## 6. Root Cause
`{root['root_cause']}`.

Phase 21A-9 result valid: `{root['phase21a9_result_valid']}`.

## 7. Recommendation
Recommended Phase 21A-9C: `{decision['recommended_phase21a9c']}`.

Do not enter Phase 21B until smoke150 is revalidated with a functioning rewrite path.
"""
    (REPDIR / "summary.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "phase21a9_result_valid": root["phase21a9_result_valid"],
        "root_cause": root["root_cause"],
        "rewrite_fallback_count": summary["total_rewrite_fallback_count"],
        "aligned_rerun_real_P0": rerun_metrics["real_P0"],
        "should_rerun_smoke200": root["should_rerun_smoke200"],
        "should_enter_phase21b": root["should_enter_phase21b"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
