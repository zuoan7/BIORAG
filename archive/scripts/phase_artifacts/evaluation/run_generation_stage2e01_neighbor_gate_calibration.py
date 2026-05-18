#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_generation_stage2c_comparison_coverage import (  # noqa: E402
    build_markdown,
    failure_category,
    load_records,
    run_group,
    build_settings as _base_build_settings,
)

FOCUS_SAMPLE_IDS = [
    "ent_015",
    "ent_026",
    "ent_064",
    "ent_007",
    "ent_020",
    "ent_084",
    "ent_090",
    "ent_094",
    "ent_021",
    "ent_092",
]


def build_settings_baseline() -> Any:
    settings = _base_build_settings("v2_stage2c_current")
    settings.generation.v2_enable_neighbor_audit = False
    return settings


def build_settings_audit() -> Any:
    settings = _base_build_settings("v2_stage2c_current")
    settings.generation.v2_enable_neighbor_audit = True
    settings.generation.v2_neighbor_window = 1
    settings.generation.v2_neighbor_promotion_dry_run = True
    settings.generation.v2_enable_neighbor_promotion = False
    settings.generation.v2_include_neighbor_context_in_qwen = False
    settings.generation.v2_neighbor_min_promotion_score = 0.05
    return settings


def _run_group_patched(group_key: str, label: str, dataset_path: Path, records: list[dict[str, Any]], build_fn: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import scripts.evaluation.run_generation_stage2c_comparison_coverage as stage2c_mod
    original_build = stage2c_mod.build_settings
    stage2c_mod.build_settings = lambda _gk: build_fn()
    try:
        summary, enriched = run_group(group_key, label, dataset_path, records)
    finally:
        stage2c_mod.build_settings = original_build
    return summary, enriched


def _support_pack_count_distribution(raw_records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for rec in raw_records:
        cnt = (rec.get("debug") or {}).get("generation_v2", {}).get("support_pack_count", 0)
        counter[str(cnt)] += 1
    return dict(sorted(counter.items(), key=lambda x: int(x[0])))


def _neighbor_audit_aggregate(raw_records: list[dict[str, Any]]) -> dict[str, Any]:
    total_candidates = 0
    total_promoted = 0
    total_excluded = 0
    total_context_only = 0
    by_status: Counter[str] = Counter()
    by_section: Counter[str] = Counter()
    score_source_dist: Counter[str] = Counter()
    promoted_by_reason: Counter[str] = Counter()
    excluded_by_reason: Counter[str] = Counter()
    context_only_by_reason: Counter[str] = Counter()
    no_support_blocked_total = 0
    potential_summary_boost = 0
    potential_comparison_boost = 0
    samples_with_audit = 0

    for rec in raw_records:
        gen_debug = ((rec.get("debug") or {}).get("generation_v2") or {})
        na = gen_debug.get("neighbor_audit") or {}
        if not na.get("enabled"):
            continue
        samples_with_audit += 1
        total_candidates += na.get("candidate_count", 0)
        total_promoted += na.get("dry_run_promoted_count", 0)
        total_excluded += na.get("excluded_count", 0)
        summary = na.get("summary") or {}
        total_context_only += summary.get("by_status", {}).get("context_only", 0)
        for status, cnt in (summary.get("by_status") or {}).items():
            by_status[status] += cnt
        for sec, cnt in (summary.get("by_section") or {}).items():
            by_section[sec] += cnt
        for src, cnt in (summary.get("score_source_distribution") or {}).items():
            score_source_dist[src] += cnt
        for reason, cnt in (summary.get("promoted_by_reason") or {}).items():
            promoted_by_reason[reason] += cnt
        for reason, cnt in (summary.get("excluded_by_reason") or {}).items():
            excluded_by_reason[reason] += cnt
        for reason, cnt in (summary.get("context_only_by_reason") or {}).items():
            context_only_by_reason[reason] += cnt
        no_support_blocked_total += summary.get("no_support_blocked_count", 0)
        potential_summary_boost += summary.get("potential_summary_boost_count", 0)
        potential_comparison_boost += summary.get("potential_comparison_boost_count", 0)

    return {
        "samples_with_audit": samples_with_audit,
        "total_candidates": total_candidates,
        "total_promoted": total_promoted,
        "total_excluded": total_excluded,
        "total_context_only": total_context_only,
        "by_status": dict(sorted(by_status.items())),
        "by_section": dict(sorted(by_section.items(), key=lambda x: -x[1])),
        "score_source_distribution": dict(sorted(score_source_dist.items())),
        "promoted_by_reason": dict(sorted(promoted_by_reason.items(), key=lambda x: -x[1])),
        "excluded_by_reason": dict(sorted(excluded_by_reason.items(), key=lambda x: -x[1])),
        "context_only_by_reason": dict(sorted(context_only_by_reason.items(), key=lambda x: -x[1])),
        "no_support_blocked_count": no_support_blocked_total,
        "potential_summary_boost_count": potential_summary_boost,
        "potential_comparison_boost_count": potential_comparison_boost,
    }


def _diff_report(baseline: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    def _diff_val(key: str) -> dict[str, Any]:
        return {"baseline": baseline.get(key), "audit": audit.get(key)}

    baseline_recs = baseline.get("raw_records") or []
    audit_recs = audit.get("raw_records") or []

    def _sp_dist(recs: list) -> dict:
        return _support_pack_count_distribution(recs)

    def _zero_cit_ids(recs: list) -> list:
        return [r.get("id") for r in recs if (r.get("citation_count") or 0) == 0 and r.get("answer_mode") not in {"refuse"}]

    def _qwen_used(recs: list) -> int:
        return sum(1 for r in recs if ((r.get("debug") or {}).get("generation_v2") or {}).get("qwen_synthesis", {}).get("used_qwen"))

    def _qwen_fallback(recs: list) -> int:
        return sum(1 for r in recs if ((r.get("debug") or {}).get("generation_v2") or {}).get("qwen_synthesis", {}).get("fallback_used"))

    changed_samples: list[dict[str, Any]] = []
    bl_by_id = {r.get("id"): r for r in baseline_recs}
    au_by_id = {r.get("id"): r for r in audit_recs}
    for sid in bl_by_id:
        bl = bl_by_id[sid]
        au = au_by_id.get(sid)
        if au is None:
            continue
        fields = ["answer_mode", "citation_count", "support_pack_count"]
        changed = {f: (bl.get(f), au.get(f)) for f in fields if bl.get(f) != au.get(f)}
        if changed:
            changed_samples.append({"id": sid, "changed_fields": changed})

    return {
        "route_match_rate": _diff_val("route_match_rate"),
        "doc_id_hit_rate": _diff_val("doc_id_hit_rate"),
        "section_hit_rate": _diff_val("section_hit_rate"),
        "answer_mode_distribution": _diff_val("answer_mode_distribution"),
        "citation_count_distribution": _diff_val("citation_count_distribution"),
        "support_pack_count_distribution": {
            "baseline": _sp_dist(baseline_recs),
            "audit": _sp_dist(audit_recs),
        },
        "zero_citation_substantive_answer_ids": {
            "baseline": _zero_cit_ids(baseline_recs),
            "audit": _zero_cit_ids(audit_recs),
        },
        "qwen_used_count": {
            "baseline": _qwen_used(baseline_recs),
            "audit": _qwen_used(audit_recs),
        },
        "qwen_fallback_count": {
            "baseline": _qwen_fallback(baseline_recs),
            "audit": _qwen_fallback(audit_recs),
        },
        "changed_samples": changed_samples,
        "neighbor_audit_aggregate": audit.get("neighbor_audit_aggregate"),
    }


def _build_focus_samples(baseline_summary: dict[str, Any], audit_summary: dict[str, Any]) -> dict[str, Any]:
    def _by_id(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {r.get("id"): r for r in (summary.get("raw_records") or [])}

    bl_by_id = _by_id(baseline_summary)
    au_by_id = _by_id(audit_summary)
    payload: dict[str, Any] = {}
    for sid in FOCUS_SAMPLE_IDS:
        payload[sid] = {}
        for label, by_id in [("baseline", bl_by_id), ("audit", au_by_id)]:
            rec = by_id.get(sid)
            if not rec:
                continue
            gen_debug = ((rec.get("debug") or {}).get("generation_v2") or {})
            na = gen_debug.get("neighbor_audit") or {}
            payload[sid][label] = {
                "id": sid,
                "question": rec.get("question"),
                "answer_mode": rec.get("answer_mode"),
                "citation_count": rec.get("citation_count"),
                "support_pack_count": gen_debug.get("support_pack_count"),
                "doc_hit": rec.get("doc_hit"),
                "section_hit": rec.get("section_hit"),
                "failure_category": rec.get("failure_category"),
                "answer_preview": rec.get("answer_preview"),
                "qwen_synthesis": gen_debug.get("qwen_synthesis") or {},
                "summary_selection": gen_debug.get("summary_selection") or {},
                "summary_plan": gen_debug.get("summary_plan") or {},
                "comparison_coverage": gen_debug.get("comparison_coverage") or {},
                "existence_guardrail": gen_debug.get("existence_guardrail") or {},
                "support_pack": gen_debug.get("support_pack") or [],
                "neighbor_audit": {
                    "enabled": na.get("enabled", False),
                    "candidate_count": na.get("candidate_count", 0),
                    "dry_run_promoted_count": na.get("dry_run_promoted_count", 0),
                    "excluded_count": na.get("excluded_count", 0),
                    "candidates": na.get("candidates") or [],
                    "by_seed": na.get("by_seed") or {},
                    "summary": na.get("summary") or {},
                },
            }
    return payload


def _build_summary_md(
    na_agg: dict[str, Any],
    diff: dict[str, Any],
    focus: dict[str, Any],
) -> str:
    lines = [
        "# Stage 2E.0.1 Neighbor Gate Calibration Summary",
        "",
        "## 1. Dry-run Isolation Check",
        "",
    ]

    changed = diff.get("changed_samples") or []
    lines.append(f"- changed_samples (any field): `{len(changed)}`")
    if changed:
        for cs in changed:
            lines.append(f"  - `{cs['id']}`: {cs['changed_fields']}")
    else:
        lines.append("- **PASS**: support_pack / citation / answer_mode completely unchanged")

    zero_bl = diff.get("zero_citation_substantive_answer_ids", {}).get("baseline") or []
    zero_au = diff.get("zero_citation_substantive_answer_ids", {}).get("audit") or []
    lines += [
        f"- zero_citation_substantive_answer_ids baseline: `{zero_bl}`",
        f"- zero_citation_substantive_answer_ids audit: `{zero_au}`",
        "",
        "## 2. Gate Calibration",
        "",
        f"- samples_with_audit: `{na_agg.get('samples_with_audit')}`",
        f"- total_candidates: `{na_agg.get('total_candidates')}`",
        f"- dry_run_promoted: `{na_agg.get('total_promoted')}`",
        f"- context_only: `{na_agg.get('total_context_only')}`",
        f"- excluded: `{na_agg.get('total_excluded')}`",
        f"- no_support_blocked_count: `{na_agg.get('no_support_blocked_count')}`",
        f"- score_source_distribution: `{json.dumps(na_agg.get('score_source_distribution', {}), ensure_ascii=False)}`",
        f"- promoted_by_reason: `{json.dumps(na_agg.get('promoted_by_reason', {}), ensure_ascii=False)}`",
        f"- excluded_by_reason: `{json.dumps(na_agg.get('excluded_by_reason', {}), ensure_ascii=False)}`",
        f"- context_only_by_reason: `{json.dumps(na_agg.get('context_only_by_reason', {}), ensure_ascii=False)}`",
        "",
        "## 3. Focus Samples",
        "",
    ]

    for sid in FOCUS_SAMPLE_IDS:
        au = focus.get(sid, {}).get("audit") or {}
        na = au.get("neighbor_audit") or {}
        candidates = na.get("candidates") or []
        promoted = [c for c in candidates if c.get("promotion_status") == "dry_run_promoted"]
        lines.append(
            f"### {sid}"
        )
        lines.append(f"- answer_mode: `{au.get('answer_mode')}` | citation_count: `{au.get('citation_count')}` | support_pack_count: `{au.get('support_pack_count')}`")
        lines.append(f"- neighbor_audit enabled: `{na.get('enabled')}` | candidates: `{na.get('candidate_count', 0)}` | promoted: `{na.get('dry_run_promoted_count', 0)}` | excluded: `{na.get('excluded_count', 0)}`")
        if promoted:
            for c in promoted:
                lines.append(
                    f"  - promoted: chunk_id=`{c.get('chunk_id')}` section=`{c.get('section')}` "
                    f"score=`{c.get('neighbor_score')}` score_source=`{c.get('source_seed_score_source')}` "
                    f"reasons=`{c.get('promotion_reasons')}`"
                )
        else:
            lines.append("  - no promoted neighbors")
        lines.append("")

    lines += [
        "## 4. Diff vs Baseline",
        "",
        f"- route_match_rate: `{diff['route_match_rate']['baseline']}` -> `{diff['route_match_rate']['audit']}`",
        f"- doc_id_hit_rate: `{diff['doc_id_hit_rate']['baseline']}` -> `{diff['doc_id_hit_rate']['audit']}`",
        f"- section_hit_rate: `{diff['section_hit_rate']['baseline']}` -> `{diff['section_hit_rate']['audit']}`",
        f"- answer_mode_distribution baseline: `{json.dumps(diff['answer_mode_distribution']['baseline'], ensure_ascii=False)}`",
        f"- answer_mode_distribution audit:    `{json.dumps(diff['answer_mode_distribution']['audit'], ensure_ascii=False)}`",
        f"- qwen_used baseline: `{diff['qwen_used_count']['baseline']}` | audit: `{diff['qwen_used_count']['audit']}`",
        f"- qwen_fallback baseline: `{diff['qwen_fallback_count']['baseline']}` | audit: `{diff['qwen_fallback_count']['audit']}`",
        "",
    ]

    return "\n".join(lines)


def main() -> int:
    dataset_path = ROOT / "data/eval/datasets/enterprise_ragas_smoke20.json"
    records = load_records(str(dataset_path))
    run_root = (
        ROOT / "reports/evaluation/ad_hoc/generation_v2_stage2e01_neighbor_gate_calibration"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_root.mkdir(parents=True, exist_ok=True)

    print("[stage2e01] Running baseline (audit disabled)...")
    baseline_summary, baseline_enriched = _run_group_patched(
        "v2_stage2d1_baseline", "v2 Stage 2E.0.1 baseline (audit off)",
        dataset_path, records, build_settings_baseline,
    )
    for item, raw in zip(baseline_enriched, baseline_summary.get("raw_records", []), strict=True):
        raw["failure_category"] = failure_category(item)
    baseline_summary["group"] = "v2_stage2d1_baseline"

    print("[stage2e01] Running neighbor audit gate calibration (dry-run)...")
    audit_summary, audit_enriched = _run_group_patched(
        "v2_stage2e01_audit", "v2 Stage 2E.0.1 neighbor audit gate calibration",
        dataset_path, records, build_settings_audit,
    )
    for item, raw in zip(audit_enriched, audit_summary.get("raw_records", []), strict=True):
        raw["failure_category"] = failure_category(item)
    audit_summary["group"] = "v2_stage2e01_audit"
    audit_summary["neighbor_audit_aggregate"] = _neighbor_audit_aggregate(audit_summary.get("raw_records") or [])

    # write individual summaries
    (run_root / "v2_stage2d1_baseline.json").write_text(
        json.dumps(baseline_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_root / "v2_stage2e01_audit.json").write_text(
        json.dumps(audit_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_root / "v2_stage2e01_audit.md").write_text(
        build_markdown(audit_summary), encoding="utf-8"
    )

    diff = _diff_report(baseline_summary, audit_summary)
    (run_root / "diff_vs_stage2d1.json").write_text(
        json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    focus = _build_focus_samples(baseline_summary, audit_summary)
    (run_root / "focus_samples.json").write_text(
        json.dumps(focus, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    na_agg = audit_summary.get("neighbor_audit_aggregate") or {}
    summary_md = _build_summary_md(na_agg, diff, focus)
    (run_root / "summary.md").write_text(summary_md, encoding="utf-8")

    print(json.dumps({
        "run_root": str(run_root.relative_to(ROOT)),
        "neighbor_audit_aggregate": na_agg,
        "diff_changed_samples": len(diff.get("changed_samples") or []),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
