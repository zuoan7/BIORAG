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


def build_settings_audit(enable_audit: bool) -> Any:
    """Stage 2E settings: v2 + comparison coverage + neighbor audit toggle."""
    settings = _base_build_settings("v2_stage2c_current")
    settings.generation.v2_enable_neighbor_audit = enable_audit
    settings.generation.v2_neighbor_window = 1
    settings.generation.v2_neighbor_promotion_dry_run = True
    settings.generation.v2_enable_neighbor_promotion = False
    settings.generation.v2_include_neighbor_context_in_qwen = False
    return settings


def _latest_stage2d1_path() -> Path:
    """Find the latest stage 2D run (used as baseline for 2E diff)."""
    for stage_dir in ("generation_v2_stage2d_summary_support",):
        base = ROOT / "reports/evaluation/ad_hoc" / stage_dir
        if not base.exists():
            continue
        for fname in ("v2_stage2d_current.json", "v2_stage2c_current.json"):
            candidates = sorted(
                p / fname for p in base.iterdir() if p.is_dir() and (p / fname).exists()
            )
            if candidates:
                return candidates[-1]
    raise FileNotFoundError("No Stage 2D baseline found under generation_v2_stage2d_summary_support/")


def _neighbor_audit_summary(raw_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate neighbor_audit debug across all samples."""
    total_candidates = 0
    total_promoted = 0
    total_excluded = 0
    by_status: Counter[str] = Counter()
    by_section: Counter[str] = Counter()
    by_distance: Counter[str] = Counter()
    potential_summary_boost = 0
    potential_comparison_boost = 0
    samples_with_audit = 0

    for record in raw_records:
        gen_debug = ((record.get("debug") or {}).get("generation_v2") or {})
        na = gen_debug.get("neighbor_audit") or {}
        if not na.get("enabled"):
            continue
        samples_with_audit += 1
        total_candidates += na.get("candidate_count", 0)
        total_promoted += na.get("dry_run_promoted_count", 0)
        total_excluded += na.get("excluded_count", 0)
        summary = na.get("summary") or {}
        for status, cnt in (summary.get("by_status") or {}).items():
            by_status[status] += cnt
        for sec, cnt in (summary.get("by_section") or {}).items():
            by_section[sec] += cnt
        potential_summary_boost += summary.get("potential_summary_boost_count", 0)
        potential_comparison_boost += summary.get("potential_comparison_boost_count", 0)
        for c in na.get("candidates") or []:
            by_distance[str(c.get("abs_distance", "?"))] += 1

    return {
        "samples_with_audit": samples_with_audit,
        "total_candidates": total_candidates,
        "total_promoted": total_promoted,
        "total_excluded": total_excluded,
        "context_only": by_status.get("context_only", 0),
        "by_status": dict(sorted(by_status.items())),
        "by_section": dict(sorted(by_section.items(), key=lambda x: -x[1])),
        "by_distance": dict(sorted(by_distance.items())),
        "potential_summary_boost_count": potential_summary_boost,
        "potential_comparison_boost_count": potential_comparison_boost,
    }


def _focus_payload(current_summary: dict[str, Any], baseline_summary: dict[str, Any]) -> dict[str, Any]:
    by_id = {
        "stage2d": {record.get("id"): record for record in baseline_summary.get("raw_records", [])},
        "stage2e": {record.get("id"): record for record in current_summary.get("raw_records", [])},
    }
    payload: dict[str, Any] = {}
    for sample_id in FOCUS_SAMPLE_IDS:
        payload[sample_id] = {}
        for label, records in by_id.items():
            record = records.get(sample_id)
            if not record:
                continue
            gen_debug = ((record.get("debug") or {}).get("generation_v2") or {})
            na = gen_debug.get("neighbor_audit") or {}
            payload[sample_id][label] = {
                "id": sample_id,
                "question": record.get("question"),
                "answer_mode": record.get("answer_mode"),
                "citation_count": record.get("citation_count"),
                "doc_hit": record.get("doc_hit"),
                "section_hit": record.get("section_hit"),
                "failure_category": record.get("failure_category"),
                "answer_preview": record.get("answer_preview"),
                "support_pack_count": gen_debug.get("support_pack_count"),
                "summary_selection": gen_debug.get("summary_selection") or {},
                "summary_plan": gen_debug.get("summary_plan") or {},
                "qwen_synthesis": gen_debug.get("qwen_synthesis") or {},
                "support_pack": gen_debug.get("support_pack") or [],
                "neighbor_audit": {
                    "enabled": na.get("enabled", False),
                    "candidate_count": na.get("candidate_count", 0),
                    "dry_run_promoted_count": na.get("dry_run_promoted_count", 0),
                    "excluded_count": na.get("excluded_count", 0),
                    "by_seed": na.get("by_seed") or {},
                    "candidates": na.get("candidates") or [],
                    "summary": na.get("summary") or {},
                },
            }
    return payload


def _diff_vs_stage2d(baseline_summary: dict[str, Any], current_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_stage2d_path": str(_latest_stage2d1_path().relative_to(ROOT)),
        "metrics": {
            "route_match_rate": {
                "stage2d": baseline_summary.get("route_match_rate"),
                "stage2e": current_summary.get("route_match_rate"),
            },
            "doc_id_hit_rate": {
                "stage2d": baseline_summary.get("doc_id_hit_rate"),
                "stage2e": current_summary.get("doc_id_hit_rate"),
            },
            "section_hit_rate": {
                "stage2d": baseline_summary.get("section_hit_rate"),
                "stage2e": current_summary.get("section_hit_rate"),
            },
            "answer_mode_distribution": {
                "stage2d": baseline_summary.get("answer_mode_distribution"),
                "stage2e": current_summary.get("answer_mode_distribution"),
            },
            "citation_count_distribution": {
                "stage2d": baseline_summary.get("citation_count_distribution"),
                "stage2e": current_summary.get("citation_count_distribution"),
            },
        },
        "neighbor_audit_aggregate": current_summary.get("neighbor_audit_aggregate"),
    }


def _run_group_with_audit(group_key: str, label: str, dataset_path: Path, records: list[dict[str, Any]], enable_audit: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Wrapper that overrides build_settings inside run_group via monkey-patching."""
    import scripts.evaluation.run_generation_stage2c_comparison_coverage as stage2c_mod
    original_build = stage2c_mod.build_settings

    def patched_build(gk: str) -> Any:
        return build_settings_audit(enable_audit)

    stage2c_mod.build_settings = patched_build
    try:
        summary, enriched = run_group(group_key, label, dataset_path, records)
    finally:
        stage2c_mod.build_settings = original_build

    return summary, enriched


def main() -> int:
    dataset_path = ROOT / "data/eval/datasets/enterprise_ragas_smoke20.json"
    records = load_records(str(dataset_path))
    run_root = ROOT / "reports/evaluation/ad_hoc/generation_v2_stage2e_neighbor_audit" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    # A. baseline: audit disabled (同 stage2d1 条件)
    print("[stage2e] Running baseline (audit disabled)...")
    baseline_summary, baseline_enriched = _run_group_with_audit(
        "v2_stage2e_baseline", "v2 Stage 2E baseline (audit off)", dataset_path, records, enable_audit=False
    )
    for item, raw in zip(baseline_enriched, baseline_summary.get("raw_records", []), strict=True):
        raw["failure_category"] = failure_category(item)
    baseline_summary["group"] = "v2_stage2d1_baseline"

    # B. neighbor audit: audit enabled, dry-run only
    print("[stage2e] Running neighbor audit (dry-run)...")
    audit_summary, audit_enriched = _run_group_with_audit(
        "v2_stage2e_audit", "v2 Stage 2E neighbor audit (dry-run)", dataset_path, records, enable_audit=True
    )
    for item, raw in zip(audit_enriched, audit_summary.get("raw_records", []), strict=True):
        raw["failure_category"] = failure_category(item)
    audit_summary["group"] = "v2_stage2e_audit"
    audit_summary["neighbor_audit_aggregate"] = _neighbor_audit_summary(audit_summary.get("raw_records", []))

    # write outputs
    (run_root / "v2_stage2d1_baseline.json").write_text(
        json.dumps(baseline_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_root / "v2_stage2e_audit.json").write_text(
        json.dumps(audit_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_root / "v2_stage2e_audit.md").write_text(
        build_markdown(audit_summary), encoding="utf-8"
    )

    diff = _diff_vs_stage2d(baseline_summary, audit_summary)
    (run_root / "diff_vs_stage2d.json").write_text(
        json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    focus = _focus_payload(audit_summary, baseline_summary)
    (run_root / "focus_samples.json").write_text(
        json.dumps(focus, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # summary md
    na_agg = audit_summary.get("neighbor_audit_aggregate") or {}
    (run_root / "summary.md").write_text(
        "\n".join([
            "# Stage 2E Neighbor Audit Summary",
            "",
            "## Neighbor Audit Aggregate",
            f"- samples_with_audit: `{na_agg.get('samples_with_audit')}`",
            f"- total_candidates: `{na_agg.get('total_candidates')}`",
            f"- dry_run_promoted: `{na_agg.get('total_promoted')}`",
            f"- context_only: `{na_agg.get('context_only')}`",
            f"- excluded: `{na_agg.get('total_excluded')}`",
            f"- potential_summary_boost_count: `{na_agg.get('potential_summary_boost_count')}`",
            f"- potential_comparison_boost_count: `{na_agg.get('potential_comparison_boost_count')}`",
            f"- by_status: `{json.dumps(na_agg.get('by_status', {}), ensure_ascii=False)}`",
            f"- by_distance: `{json.dumps(na_agg.get('by_distance', {}), ensure_ascii=False)}`",
            "",
            "## Focus Samples: neighbor_audit",
            "",
            *[
                f"- `{sid}` candidates=`{focus.get(sid, {}).get('stage2e', {}).get('neighbor_audit', {}).get('candidate_count', 0)}` "
                f"promoted=`{focus.get(sid, {}).get('stage2e', {}).get('neighbor_audit', {}).get('dry_run_promoted_count', 0)}` "
                f"support_pack_count=`{focus.get(sid, {}).get('stage2e', {}).get('support_pack_count', '?')}`"
                for sid in FOCUS_SAMPLE_IDS
            ],
            "",
            "## Diff vs Stage 2D",
            f"- route_match_rate: `{diff['metrics']['route_match_rate']['stage2d']}` -> `{diff['metrics']['route_match_rate']['stage2e']}`",
            f"- doc_id_hit_rate: `{diff['metrics']['doc_id_hit_rate']['stage2d']}` -> `{diff['metrics']['doc_id_hit_rate']['stage2e']}`",
            f"- answer_mode_distribution: `{json.dumps(diff['metrics']['answer_mode_distribution']['stage2d'], ensure_ascii=False)}` -> `{json.dumps(diff['metrics']['answer_mode_distribution']['stage2e'], ensure_ascii=False)}`",
            "",
        ]),
        encoding="utf-8",
    )

    print(json.dumps({"run_root": str(run_root.relative_to(ROOT)), "neighbor_audit_aggregate": na_agg}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
