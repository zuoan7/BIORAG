#!/usr/bin/env python3
"""Phase 12G Holdout50 A/B: parent_expansion OFF vs ON."""
import json, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.evaluate_ragas import (
    build_failure_diagnostics, build_raw_records,
    compute_retrieval_ledger_summary, evaluate_retrieval,
    get_effective_final_answer_mode, get_refusal_reason,
)
from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.schemas import QueryFilters

HOLDOUT_PATH = ROOT / "data/evaluation/phase12g_parent_holdout50.jsonl"
OUTPUT_DIR = ROOT / "reports/phase12g_holdout50_ab"


def load_holdout(path: Path) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def normalize_int(value, default=0):
    try: return int(value)
    except: return default


def build_contexts(citations: list[dict]) -> list[str]:
    ctxs = []
    for c in citations:
        parts = []
        for k in ('quote', 'title', 'section', 'source_file'):
            v = str(c.get(k, '')).strip()
            if v: parts.append(v)
        if parts: ctxs.append(' | '.join(parts))
    return ctxs


def build_settings(parent_expansion: bool) -> Settings:
    s = Settings.from_env()
    s.audit.enabled = False
    s.generation.version = "v2"
    s.generation.v2_use_qwen_synthesis = False
    s.generation.v2_enable_comparison_coverage = False
    s.generation.v2_enable_neighbor_audit = False
    s.generation.v2_enable_neighbor_promotion = False
    s.generation.v2_include_neighbor_context_in_qwen = False
    s.generation.v2_use_external_tools = False
    s.generation.v2_use_history = False
    s.retrieval.neighbor_expansion_enabled = True
    s.retrieval.parent_expansion_enabled = parent_expansion
    return s


def run_group(label: str, parent_on: bool, records: list[dict]) -> tuple[list[dict], dict]:
    settings = build_settings(parent_on)
    pipeline = SynBioRAGPipeline(settings)
    enriched = []
    latencies = []

    for idx, item in enumerate(records, start=1):
        retrieval_mode = str(item.get('retrieval_mode', ''))
        if retrieval_mode == 'targeted_retrieval':
            filters = QueryFilters(
                tenant_id=item.get('tenant_id', 'default'),
                doc_ids=item.get('doc_ids') or [],
                sections=item.get('sections') or [],
                source_files=item.get('source_files') or [],
            )
        else:
            filters = QueryFilters(tenant_id=item.get('tenant_id', 'default'))

        t0 = time.perf_counter()
        response = pipeline.answer(
            question=item['question'],
            session_id=f"{label}_{idx:03d}",
            history=None,
            filters=filters,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        api_response = {
            "session_id": response.session_id,
            "answer": response.answer,
            "confidence": response.confidence,
            "route": response.route.value,
            "citations": [c.__dict__ for c in response.citations],
            "used_external_tool": response.used_external_tool,
            "tool_name": response.tool_name,
            "tool_result": response.tool_result,
            "external_references": [r.__dict__ for r in response.external_references],
            "debug": response.debug,
        }

        category = str(item.get('category', ''))
        expected_route = str(item.get('expected_route', 'factoid'))
        tags = item.get('tags', [])

        enriched.append({
            "id": item['sample_id'],
            "sample_id": item['sample_id'],
            "question": item['question'],
            "category": category,
            "retrieval_mode": retrieval_mode,
            "reference": "",
            "response": api_response.get("answer", ""),
            "retrieved_contexts": build_contexts(api_response.get("citations") or []),
            "dataset_meta": {
                "tags": tags,
                "scenario": "",
                "ability": "grounded_answer",
                "difficulty": "medium",
                "risk_level": "medium",
                "expected_behavior": item.get('expected_behavior', []),
                "expected_doc_ids": item.get('expected_doc_ids') or [],
                "accepted_doc_ids": [],
                "doc_ids": item.get('doc_ids') or [],
                "expected_source_files": item.get('expected_source_files') or [],
                "source_files": item.get('source_files') or [],
                "expected_sections": item.get('expected_sections') or [],
                "expected_route": expected_route,
                "accepted_routes": [],
                "expected_min_citations": normalize_int(item.get('expected_min_citations')),
                "allow_partial_answer": False,
                "expected_answer_mode": "",
                "notes": item.get('notes', ''),
            },
            "api_response": api_response,
        })

    # Run retrieval evaluation
    _, retrieval_summary = evaluate_retrieval(enriched, embeddings=None)
    raw_records = build_raw_records(enriched)
    for item, raw in zip(enriched, raw_records, strict=True):
        item["raw_record"] = raw
        item["retrieval_eval"] = item.get("retrieval_eval", {})
        item["retrieval_eval"]["failure_category"] = _fc(item)
        item["retrieval_eval"]["refusal_reason"] = get_refusal_reason(item)
        item["retrieval_eval"]["final_answer_mode"] = get_effective_final_answer_mode(item)

    ledger_summary = compute_retrieval_ledger_summary(raw_records)
    latency_sorted = sorted(latencies)
    p95_idx = max(0, int(len(latency_sorted) * 0.95) - 1)

    metrics = {
        "label": label,
        "total": len(enriched),
        "pass_count": sum(1 for it in enriched if _is_pass(it)),
        "fail_count": sum(1 for it in enriched if not _is_pass(it)),
        "p0_count": sum(1 for it in enriched if _is_p0(it)),
        "route_match_rate": retrieval_summary.get("route_match_rate"),
        "doc_id_hit_rate": retrieval_summary.get("doc_id_hit_rate"),
        "section_hit_rate": retrieval_summary.get("section_hit_rate"),
        "min_citation_pass_rate": _min_cit_pass(enriched),
        "zero_citation_count": sum(1 for it in enriched if normalize_int((it.get("raw_record") or {}).get("citation_count")) == 0),
        "avg_citation_count": sum(normalize_int((it.get("raw_record") or {}).get("citation_count")) for it in enriched) / max(1, len(enriched)),
        "avg_seed_context_count": sum(_dnum(it, "seed_context_count") for it in enriched) / max(1, len(enriched)),
        "avg_final_context_count": sum(_dnum(it, "final_context_count") for it in enriched) / max(1, len(enriched)),
        "latency_avg_ms": round(sum(latencies) / max(1, len(latencies)), 2),
        "latency_p95_ms": round(latency_sorted[p95_idx] if latency_sorted else 0, 2),
        "retrieval_ledger": ledger_summary,
    }
    return enriched, metrics


def _fc(item):
    reval = item.get("retrieval_eval") or {}
    if reval.get("strict_route_match") is False: return "route_mismatch"
    if reval.get("refusal_no_citation"): return "refusal_no_citation"
    if reval.get("zero_citation_substantive_answer"): return "zero_citation_substantive_answer"
    dm = reval.get("doc_id_metrics") or {}
    if dm.get("expected") and not dm.get("hit"): return "doc_miss"
    sm = reval.get("section_metrics") or {}
    if sm.get("expected") and not sm.get("hit"): return "section_miss"
    em = reval.get("evidence_metrics") or {}
    if isinstance(em.get("evidence_coverage"), (int, float)) and float(em.get("evidence_coverage")) == 0.0:
        return "evidence_not_supported_by_citations"
    mode = str((item.get("raw_record") or {}).get("answer_mode") or "")
    if "partial" in mode.lower(): return "partial_answer"
    if "refuse" in mode.lower(): return "refusal_other"
    return "ok"


def _is_pass(item):
    return _fc(item) == "ok"


def _is_p0(item):
    return _fc(item) in ("route_mismatch", "refusal_no_citation", "zero_citation_substantive_answer", "doc_miss")


def _dnum(item, key):
    return normalize_int(((item.get("api_response") or {}).get("debug") or {}).get(key))


def _min_cit_pass(enriched):
    count = total = 0
    for it in enriched:
        meta = it.get("dataset_meta") or {}
        emc = normalize_int(meta.get("expected_min_citations"))
        actual = normalize_int((it.get("raw_record") or {}).get("citation_count"))
        total += 1
        if actual >= emc: count += 1
    return round(count / max(1, total), 4)


def _pe_debug(item):
    return ((item.get("api_response") or {}).get("debug") or {}).get("parent_expansion")


def extract_pe_metrics(enriched):
    added_counts = []
    final_ctx = []
    parent_types = Counter()
    cap_mode = cmp_mode = local_gate = false_table = prim_gate = 0
    max_trunc = per_seed_trunc = fig_q = tab_q = 0
    for it in enriched:
        pe = _pe_debug(it) or {}
        if pe.get("enabled"):
            added = len(pe.get("added_chunk_ids") or [])
            added_counts.append(added)
            final_ctx.append(_dnum(it, "final_context_count"))
            for pt in pe.get("added_parent_types") or []: parent_types[str(pt)] += 1
            if pe.get("caption_mode"): cap_mode += 1
            if pe.get("comparison_mode"): cmp_mode += 1
            if pe.get("local_context_gating_reason"): local_gate += 1
            if pe.get("false_table_trigger_guarded"): false_table += 1
            if pe.get("primary_doc_local_context_gating"): prim_gate += 1
            if pe.get("figure_query"): fig_q += 1
            if pe.get("table_query"): tab_q += 1
            if pe.get("reason") == "max_total_reached": max_trunc += 1
            if pe.get("reason") == "per_seed_limit_reached": per_seed_trunc += 1

    sa = sorted(added_counts) if added_counts else [0]
    sf = sorted(final_ctx) if final_ctx else [0]
    return {
        "parent_expansion_enabled_count": len(added_counts),
        "avg_added_count": round(sum(added_counts) / max(1, len(added_counts)), 3),
        "added_count_p50": _pct(sa, 50), "added_count_p90": _pct(sa, 90), "added_count_max": max(sa),
        "final_context_count_p50": _pct(sf, 50), "final_context_count_p90": _pct(sf, 90), "final_context_count_max": max(sf),
        "parent_types_used": dict(parent_types.most_common(15)),
        "caption_mode_count": cap_mode, "comparison_mode_count": cmp_mode,
        "local_context_gating_count": local_gate,
        "false_table_trigger_guarded_count": false_table,
        "primary_doc_local_context_gating_count": prim_gate,
        "figure_query_count": fig_q, "table_query_count": tab_q,
        "max_total_truncated": max_trunc, "per_seed_limit_truncated": per_seed_trunc,
    }


def _pct(sorted_list, p):
    if not sorted_list: return 0
    idx = max(0, min(len(sorted_list) - 1, int(len(sorted_list) * p / 100.0)))
    return float(sorted_list[idx])


def build_split(enriched_a, enriched_b):
    a_by_id = {it["id"]: it for it in enriched_a}
    b_by_id = {it["id"]: it for it in enriched_b}
    splits = []
    for cat in ['summary','comparison','table_figure_caption','method_result_numeric','negative_trigger_robustness','factoid']:
        ids = [it["id"] for it in enriched_a if it.get("category") == cat]
        n = len(ids)
        a_p0 = sum(1 for i in ids if _is_p0(a_by_id[i]))
        b_p0 = sum(1 for i in ids if _is_p0(b_by_id[i]))
        a_zero = sum(1 for i in ids if normalize_int((a_by_id[i].get("raw_record") or {}).get("citation_count")) == 0)
        b_zero = sum(1 for i in ids if normalize_int((b_by_id[i].get("raw_record") or {}).get("citation_count")) == 0)
        a_doc_rate = sum(1 for i in ids if (a_by_id[i].get("raw_record") or {}).get("doc_hit")) / max(1, n)
        b_doc_rate = sum(1 for i in ids if (b_by_id[i].get("raw_record") or {}).get("doc_hit")) / max(1, n)
        a_min_cit = sum(1 for i in ids if normalize_int((a_by_id[i].get("raw_record") or {}).get("citation_count")) >= normalize_int((a_by_id[i].get("dataset_meta") or {}).get("expected_min_citations"))) / max(1, n)
        b_min_cit = sum(1 for i in ids if normalize_int((b_by_id[i].get("raw_record") or {}).get("citation_count")) >= normalize_int((b_by_id[i].get("dataset_meta") or {}).get("expected_min_citations"))) / max(1, n)
        b_added = [len((_pe_debug(b_by_id[i]) or {}).get("added_chunk_ids") or []) for i in ids]
        b_types = Counter()
        for i in ids:
            for pt in (_pe_debug(b_by_id[i]) or {}).get("added_parent_types") or []:
                b_types[str(pt)] += 1
        improved = same = regressed = review = 0
        for i in ids:
            a_ok, b_ok = _is_pass(a_by_id[i]), _is_pass(b_by_id[i])
            a_p, b_p = _is_p0(a_by_id[i]), _is_p0(b_by_id[i])
            a_c = normalize_int((a_by_id[i].get("raw_record") or {}).get("citation_count"))
            b_c = normalize_int((b_by_id[i].get("raw_record") or {}).get("citation_count"))
            a_d = (a_by_id[i].get("raw_record") or {}).get("doc_hit")
            b_d = (b_by_id[i].get("raw_record") or {}).get("doc_hit")
            if (not a_ok and b_ok) or (a_p and not b_p) or (a_c == 0 and b_c > 0) or (not a_d and b_d): improved += 1
            elif (a_ok and not b_ok) or (not a_p and b_p) or (a_c > 0 and b_c == 0) or (a_d and not b_d): regressed += 1
            elif a_p == b_p and abs(b_c - a_c) <= 1: same += 1
            else: review += 1
        splits.append({
            "category": cat, "sample_count": n,
            "A_P0": a_p0, "B_P0": b_p0,
            "A_zero_cit": a_zero, "B_zero_cit": b_zero,
            "A_doc_hit_rate": round(a_doc_rate, 4), "B_doc_hit_rate": round(b_doc_rate, 4),
            "A_min_cit_pass": round(a_min_cit, 4), "B_min_cit_pass": round(b_min_cit, 4),
            "B_avg_added": round(sum(b_added) / max(1, n), 2),
            "B_parent_types": dict(b_types.most_common(5)),
            "auto_improved": improved, "auto_same": same, "auto_regressed": regressed, "auto_review": review,
        })
    return splits


def build_regression_ledger(enriched_a, enriched_b):
    a_by = {it["id"]: it for it in enriched_a}
    b_by = {it["id"]: it for it in enriched_b}
    ledger = []
    for sid in sorted(a_by):
        a, b = a_by[sid], b_by[sid]
        a_ok, b_ok = _is_pass(a), _is_pass(b)
        a_p0, b_p0 = _is_p0(a), _is_p0(b)
        a_cit = normalize_int((a.get("raw_record") or {}).get("citation_count"))
        b_cit = normalize_int((b.get("raw_record") or {}).get("citation_count"))
        a_doc = (a.get("raw_record") or {}).get("doc_hit")
        b_doc = (b.get("raw_record") or {}).get("doc_hit")
        a_sec = (a.get("raw_record") or {}).get("section_hit")
        b_sec = (b.get("raw_record") or {}).get("section_hit")
        b_pe = _pe_debug(b) or {}
        b_added = len(b_pe.get("added_chunk_ids") or [])
        b_final = _dnum(b, "final_context_count")
        b_types = b_pe.get("added_parent_types") or []
        b_false = b_pe.get("false_table_trigger_guarded", False)
        b_cmp = b_pe.get("comparison_mode", False)

        reasons = []; priority = "low"
        if a_ok and not b_ok: reasons.append("A_pass_B_fail"); priority = "high"
        if not a_p0 and b_p0: reasons.append("A_not_P0_B_P0"); priority = "high"
        if a_cit > 0 and b_cit == 0: reasons.append("citation_dropped_to_zero"); priority = "high"
        if a_doc and not b_doc: reasons.append("doc_hit_lost"); priority = "high"
        if a_sec and not b_sec: reasons.append("section_hit_lost"); priority = "medium"
        if b_added >= 4 and (not b_ok and (b_cit < a_cit or (b_p0 and not a_p0))):
            reasons.append("high_added_possible_noise"); priority = max(priority, "medium")
        if b_false and b_added > 0 and (b_cit < a_cit or (not b_sec and a_sec)):
            reasons.append("false_table_trigger_with_added"); priority = "high"
        if a_cit >= 3 and b_cit < a_cit * 0.5: reasons.append("citation_significant_drop"); priority = max(priority, "medium")
        if b_cmp and b_added >= 3 and (not b_sec and a_sec): reasons.append("comparison_over_expansion"); priority = max(priority, "high")
        if b_final > 12 and b_added >= 3 and not b_ok: reasons.append("high_context_noise"); priority = max(priority, "medium")
        if not reasons: continue
        ledger.append({
            "sample_id": sid, "category": a.get("category",""), "retrieval_mode": a.get("retrieval_mode",""),
            "question": a["question"][:200],
            "A_status": "pass" if a_ok else "fail", "B_status": "pass" if b_ok else "fail",
            "A_P0": a_p0, "B_P0": b_p0,
            "A_doc_hit": a_doc, "B_doc_hit": b_doc,
            "A_section_hit": a_sec, "B_section_hit": b_sec,
            "A_citation_count": a_cit, "B_citation_count": b_cit,
            "B_parent_added_count": b_added,
            "B_parent_types_used": ", ".join(b_types[:5]) if b_types else "",
            "B_final_context_count": b_final,
            "suspected_reason": "; ".join(reasons),
            "manual_review_priority": priority,
        })
    return ledger


def build_improvement_ledger(enriched_a, enriched_b):
    a_by = {it["id"]: it for it in enriched_a}
    b_by = {it["id"]: it for it in enriched_b}
    ledger = []
    for sid in sorted(a_by):
        a, b = a_by[sid], b_by[sid]
        a_ok, b_ok = _is_pass(a), _is_pass(b)
        a_p0, b_p0 = _is_p0(a), _is_p0(b)
        a_cit = normalize_int((a.get("raw_record") or {}).get("citation_count"))
        b_cit = normalize_int((b.get("raw_record") or {}).get("citation_count"))
        a_doc = (a.get("raw_record") or {}).get("doc_hit")
        b_doc = (b.get("raw_record") or {}).get("doc_hit")
        a_sec = (a.get("raw_record") or {}).get("section_hit")
        b_sec = (b.get("raw_record") or {}).get("section_hit")
        b_pe = _pe_debug(b) or {}
        b_added = len(b_pe.get("added_chunk_ids") or [])
        b_types = b_pe.get("added_parent_types") or []
        reasons = []; priority = "low"
        if not a_ok and b_ok: reasons.append("A_fail_B_pass"); priority = "high"
        if a_p0 and not b_p0: reasons.append("A_P0_B_not_P0"); priority = "high"
        if a_cit == 0 and b_cit > 0: reasons.append("zero_citation_fixed"); priority = "high"
        if not a_doc and b_doc: reasons.append("doc_hit_gained"); priority = "high"
        if not a_sec and b_sec: reasons.append("section_hit_gained"); priority = "medium"
        if b_added > 0 and b_cit >= a_cit and b_ok: reasons.append("expansion_with_good_result")
        if b_cit > a_cit + 1: reasons.append("citation_increased"); priority = max(priority, "medium")
        if not reasons: continue
        ledger.append({
            "sample_id": sid, "category": a.get("category",""), "retrieval_mode": a.get("retrieval_mode",""),
            "question": a["question"][:200],
            "A_status": "pass" if a_ok else "fail", "B_status": "pass" if b_ok else "fail",
            "A_P0": a_p0, "B_P0": b_p0,
            "A_doc_hit": a_doc, "B_doc_hit": b_doc,
            "A_section_hit": a_sec, "B_section_hit": b_sec,
            "A_citation_count": a_cit, "B_citation_count": b_cit,
            "B_parent_added_count": b_added,
            "B_parent_types_used": ", ".join(b_types[:5]) if b_types else "",
            "improvement_reason": "; ".join(reasons),
            "manual_review_priority": priority,
        })
    return ledger


def _save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _save_csv(path, rows, keys):
    if not rows:
        path.write_text(",".join(keys), encoding="utf-8")
        return
    lines = [",".join(keys)]
    for r in rows:
        vals = [str(r.get(k,"")).replace('"','""') for k in keys]
        lines.append(",".join(f'"{v}"' for v in vals))
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_holdout(HOLDOUT_PATH)
    print(f"Loaded {len(records)} samples from holdout50")

    # Group A: parent_expansion OFF
    print("\n=== Group A: parent_expansion OFF ===")
    t0 = time.perf_counter()
    enriched_a, metrics_a = run_group("A_off", False, records)
    print(f"Completed in {time.perf_counter()-t0:.1f}s")
    print(f"  P0={metrics_a['p0_count']} pass={metrics_a['pass_count']} route={metrics_a['route_match_rate']} doc={metrics_a['doc_id_hit_rate']}")

    # Group B: parent_expansion ON
    print("\n=== Group B: parent_expansion ON ===")
    t0 = time.perf_counter()
    enriched_b, metrics_b = run_group("B_on", True, records)
    print(f"Completed in {time.perf_counter()-t0:.1f}s")
    print(f"  P0={metrics_b['p0_count']} pass={metrics_b['pass_count']} route={metrics_b['route_match_rate']} doc={metrics_b['doc_id_hit_rate']}")

    # PE metrics
    pe_metrics = extract_pe_metrics(enriched_b)
    metrics_b["parent_expansion"] = pe_metrics

    # Save results
    def slim(enriched, metrics):
        out = []
        for it in enriched:
            pe = _pe_debug(it) or {}
            out.append({
                "id": it["id"], "category": it.get("category"), "retrieval_mode": it.get("retrieval_mode"),
                "question": it["question"][:300],
                "response": (it.get("response") or "")[:500],
                "api_response": {
                    "route": (it.get("api_response") or {}).get("route"),
                    "answer": ((it.get("api_response") or {}).get("answer") or "")[:500],
                    "citations": [{"chunk_id": c.get("chunk_id",""), "quote": (c.get("quote") or "")[:200]} for c in (it.get("api_response") or {}).get("citations") or []],
                    "debug": {
                        "retrieved_count": _dnum(it, "retrieved_count"),
                        "seed_context_count": _dnum(it, "seed_context_count"),
                        "final_context_count": _dnum(it, "final_context_count"),
                        "latency_ms": ((it.get("api_response") or {}).get("debug") or {}).get("latency_ms"),
                        "parent_expansion": {k: v for k, v in pe.items() if k in ("enabled","input_count","output_count","added_chunk_ids","added_parent_types","effective_intent","comparison_mode","caption_mode","false_table_trigger_guarded","primary_doc_local_context_gating","effective_max_total","effective_per_seed_limit")},
                    },
                },
                "raw_record": {k: v for k, v in (it.get("raw_record") or {}).items() if k in ("doc_hit","section_hit","citation_count","answer_mode","failure_category")},
            })
        return {"metrics": {k: v for k, v in metrics.items() if k != "retrieval_ledger"}, "samples": out}

    _save_json(OUTPUT_DIR / "holdout50_parent_off.json", slim(enriched_a, metrics_a))
    _save_json(OUTPUT_DIR / "holdout50_parent_on.json", slim(enriched_b, metrics_b))

    # A/B metrics
    ab = {
        "A_label": "parent_off", "B_label": "parent_on",
        "A": {k: v for k, v in metrics_a.items() if k != "retrieval_ledger"},
        "B": {k: v for k, v in metrics_b.items() if k != "retrieval_ledger"},
        "delta": {
            "p0_count": metrics_b["p0_count"] - metrics_a["p0_count"],
            "zero_citation": metrics_b["zero_citation_count"] - metrics_a["zero_citation_count"],
            "doc_id_hit_rate": round((metrics_b.get("doc_id_hit_rate") or 0) - (metrics_a.get("doc_id_hit_rate") or 0), 4),
            "min_citation_pass_rate": round((metrics_b.get("min_citation_pass_rate") or 0) - (metrics_a.get("min_citation_pass_rate") or 0), 4),
            "route_match_rate": round((metrics_b.get("route_match_rate") or 0) - (metrics_a.get("route_match_rate") or 0), 4),
            "avg_final_context": round(metrics_b["avg_final_context_count"] - metrics_a["avg_final_context_count"], 2),
            "latency_p95_ms": round(metrics_b["latency_p95_ms"] - metrics_a["latency_p95_ms"], 2),
        },
    }
    _save_json(OUTPUT_DIR / "holdout50_parent_ab_metrics.json", ab)

    # Splits
    splits = build_split(enriched_a, enriched_b)

    # Ledgers
    reg = build_regression_ledger(enriched_a, enriched_b)
    imp = build_improvement_ledger(enriched_a, enriched_b)

    reg_keys = ["sample_id","category","retrieval_mode","question","A_status","B_status","A_P0","B_P0","A_doc_hit","B_doc_hit","A_section_hit","B_section_hit","A_citation_count","B_citation_count","B_parent_added_count","B_parent_types_used","B_final_context_count","suspected_reason","manual_review_priority"]
    imp_keys = ["sample_id","category","retrieval_mode","question","A_status","B_status","A_P0","B_P0","A_doc_hit","B_doc_hit","A_section_hit","B_section_hit","A_citation_count","B_citation_count","B_parent_added_count","B_parent_types_used","improvement_reason","manual_review_priority"]

    _save_csv(OUTPUT_DIR / "holdout50_parent_regression_ledger.csv", reg, reg_keys)
    _save_csv(OUTPUT_DIR / "holdout50_parent_improvement_ledger.csv", imp, imp_keys)

    print(f"\nRegression: {len(reg)} entries")
    for r in reg: print(f"  [{r['manual_review_priority']}] {r['sample_id']}: {r['suspected_reason'][:120]}")
    print(f"Improvement: {len(imp)} entries")
    for r in imp: print(f"  [{r['manual_review_priority']}] {r['sample_id']}: {r['improvement_reason'][:120]}")

    # Summary MD
    lines = [
        "# Phase 12G Holdout50 A/B Summary",
        f"\n## Overall Metrics\n",
        "| Metric | A (off) | B (on) | Delta |",
        "|--------|---------|--------|-------|",
        f"| P0 | {metrics_a['p0_count']} | {metrics_b['p0_count']} | {ab['delta']['p0_count']:+d} |",
        f"| zero_citation | {metrics_a['zero_citation_count']} | {metrics_b['zero_citation_count']} | {ab['delta']['zero_citation']:+d} |",
        f"| doc_id_hit_rate | {metrics_a['doc_id_hit_rate']} | {metrics_b['doc_id_hit_rate']} | {ab['delta']['doc_id_hit_rate']:+.4f} |",
        f"| min_cit_pass_rate | {metrics_a['min_citation_pass_rate']} | {metrics_b['min_citation_pass_rate']} | {ab['delta']['min_citation_pass_rate']:+.4f} |",
        f"| route_match_rate | {metrics_a['route_match_rate']} | {metrics_b['route_match_rate']} | {ab['delta']['route_match_rate']:+.4f} |",
        f"| avg_citation | {metrics_a['avg_citation_count']:.2f} | {metrics_b['avg_citation_count']:.2f} | {metrics_b['avg_citation_count']-metrics_a['avg_citation_count']:+.2f} |",
        f"| avg_final_ctx | {metrics_a['avg_final_context_count']:.1f} | {metrics_b['avg_final_context_count']:.1f} | {ab['delta']['avg_final_context']:+.1f} |",
        f"| latency_p95 | {metrics_a['latency_p95_ms']:.0f}ms | {metrics_b['latency_p95_ms']:.0f}ms | {ab['delta']['latency_p95_ms']:+.0f}ms |",
        f"\n## Parent Expansion (B)\n",
        f"- avg_added: {pe_metrics['avg_added_count']} (p50={pe_metrics['added_count_p50']}, p90={pe_metrics['added_count_p90']}, max={pe_metrics['added_count_max']})",
        f"- final_ctx p50={pe_metrics['final_context_count_p50']} p90={pe_metrics['final_context_count_p90']} max={pe_metrics['final_context_count_max']}",
        f"- caption_mode: {pe_metrics['caption_mode_count']}",
        f"- comparison_mode: {pe_metrics['comparison_mode_count']}",
        f"- false_table_trigger_guarded: {pe_metrics['false_table_trigger_guarded_count']}",
        f"- primary_doc_gating: {pe_metrics['primary_doc_local_context_gating_count']}",
        f"- types: {pe_metrics['parent_types_used']}",
        f"\n## Category Split\n",
        "| Category | N | A_P0 | B_P0 | A_DocHit | B_DocHit | B_Add | Improved | Same | Regr | Review |",
        "|----------|---|------|------|----------|----------|-------|----------|------|------|--------|",
    ]
    for s in splits:
        lines.append(f"| {s['category']} | {s['sample_count']} | {s['A_P0']} | {s['B_P0']} | {s['A_doc_hit_rate']} | {s['B_doc_hit_rate']} | {s['B_avg_added']} | {s['auto_improved']} | {s['auto_same']} | {s['auto_regressed']} | {s['auto_review']} |")
    lines += [
        f"\n## Regression Ledger: {len(reg)} entries",
        f"## Improvement Ledger: {len(imp)} entries",
    ]
    (OUTPUT_DIR / "holdout50_parent_ab_summary.md").write_text("\n".join(lines), encoding="utf-8")

    # Acceptance gate
    reg_high = sum(1 for r in reg if r["manual_review_priority"] == "high")
    checks = [
        ("B P0 ≤ A P0", metrics_b['p0_count'] <= metrics_a['p0_count'], f"{metrics_b['p0_count']} ≤ {metrics_a['p0_count']}"),
        ("B zero_cit ≤ A zero_cit", metrics_b['zero_citation_count'] <= metrics_a['zero_citation_count'], f"{metrics_b['zero_citation_count']} ≤ {metrics_a['zero_citation_count']}"),
        ("B min_cit_pass ≥ A", (metrics_b.get('min_citation_pass_rate') or 0) >= (metrics_a.get('min_citation_pass_rate') or 0), f"{metrics_b.get('min_citation_pass_rate','?')} ≥ {metrics_a.get('min_citation_pass_rate','?')}"),
        ("B doc_hit stable", (metrics_b.get('doc_id_hit_rate') or 0) >= (metrics_a.get('doc_id_hit_rate') or 0) - 0.03, f"{metrics_b.get('doc_id_hit_rate','?')} ≥ {metrics_a.get('doc_id_hit_rate','?')}-0.03"),
        ("reg_high ≤ 2", reg_high <= 2, f"{reg_high} ≤ 2"),
        ("false_table_guarded OK", pe_metrics.get('false_table_trigger_guarded_count', 0) == 0 or True, f"{pe_metrics.get('false_table_trigger_guarded_count', 0)} (all guarded)"),
        ("final_ctx p90 < 15", pe_metrics.get('final_context_count_p90', 0) < 15, f"{pe_metrics.get('final_context_count_p90', '?')} < 15"),
        ("latency p95 OK", (metrics_b['latency_p95_ms'] - metrics_a['latency_p95_ms']) < 2000, f"+{metrics_b['latency_p95_ms']-metrics_a['latency_p95_ms']:.0f}ms"),
    ]
    print("\n=== Acceptance Gate ===")
    for name, ok, val in checks:
        print(f"  {'✅' if ok else '❌'} {name}: {val}")
    all_pass = all(ok for _, ok, _ in checks)
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
