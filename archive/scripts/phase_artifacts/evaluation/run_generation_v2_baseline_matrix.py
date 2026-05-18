#!/usr/bin/env python3
"""
Generation V2 ablation baseline matrix.

Runs up to 6 configuration groups against enterprise_ragas_smoke20 and writes
a comparison summary to reports/evaluation/ad_hoc/generation_v2_baseline_matrix/<timestamp>/.

Groups
------
  old_baseline               GENERATION_VERSION=old  (legacy pipeline)
  v2_extractive_only         v2, no Qwen, no comparison_coverage, no neighbor_audit
  v2_qwen                    v2 + Qwen synthesis
  v2_qwen_comparison         v2 + Qwen + comparison_coverage
  v2_qwen_comparison_summary v2 + Qwen + comparison_coverage  (alias, current default state)
  v2_qwen_comparison_neighbor_audit  v2 + Qwen + comparison_coverage + neighbor_audit (dry-run)

Usage
-----
  python scripts/evaluation/run_generation_v2_baseline_matrix.py
  python scripts/evaluation/run_generation_v2_baseline_matrix.py --groups v2_extractive_only v2_qwen
  python scripts/evaluation/run_generation_v2_baseline_matrix.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.evaluation.run_generation_stage2c_comparison_coverage as _stage2c  # noqa: E402
from scripts.evaluation.run_generation_stage2c_comparison_coverage import (  # noqa: E402
    build_markdown,
    failure_category,
    load_records,
    run_group,
)
from src.synbio_rag.domain.config import Settings  # noqa: E402

DATASET_PATH = ROOT / "data/eval/datasets/enterprise_ragas_smoke20.json"

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

# ---------------------------------------------------------------------------
# Settings builders – one per ablation group
# ---------------------------------------------------------------------------

def _base_v2() -> Settings:
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
    return s


def _build_old_baseline() -> Settings:
    s = Settings.from_env()
    s.audit.enabled = False
    s.generation.version = "old"
    return s


def _build_v2_extractive_only() -> Settings:
    return _base_v2()


def _build_v2_qwen() -> Settings:
    s = _base_v2()
    s.generation.v2_use_qwen_synthesis = True
    return s


def _build_v2_qwen_comparison() -> Settings:
    s = _base_v2()
    s.generation.v2_use_qwen_synthesis = True
    s.generation.v2_enable_comparison_coverage = True
    return s


def _build_v2_qwen_comparison_neighbor_audit() -> Settings:
    s = _build_v2_qwen_comparison()
    s.generation.v2_enable_neighbor_audit = True
    s.generation.v2_neighbor_window = 1
    s.generation.v2_neighbor_promotion_dry_run = True
    return s


ALL_GROUPS: dict[str, tuple[str, Callable[[], Settings]]] = {
    "old_baseline": (
        "v2 ablation: old baseline (legacy pipeline)",
        _build_old_baseline,
    ),
    "v2_extractive_only": (
        "v2 ablation: extractive only (no Qwen, no comparison_coverage)",
        _build_v2_extractive_only,
    ),
    "v2_qwen": (
        "v2 ablation: v2 + Qwen synthesis",
        _build_v2_qwen,
    ),
    "v2_qwen_comparison": (
        "v2 ablation: v2 + Qwen + comparison_coverage",
        _build_v2_qwen_comparison,
    ),
    "v2_qwen_comparison_summary": (
        "v2 ablation: v2 + Qwen + comparison_coverage (current default state alias)",
        _build_v2_qwen_comparison,
    ),
    "v2_qwen_comparison_neighbor_audit": (
        "v2 ablation: v2 + Qwen + comparison_coverage + neighbor_audit (dry-run)",
        _build_v2_qwen_comparison_neighbor_audit,
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_group_with_settings(
    group_key: str, label: str, build_fn: Callable[[], Settings]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run a single group by monkey-patching stage2c.build_settings."""
    original = _stage2c.build_settings
    _stage2c.build_settings = lambda _gk: build_fn()
    try:
        summary, enriched = run_group(group_key, label, DATASET_PATH, _records)
    finally:
        _stage2c.build_settings = original
    for item, raw in zip(enriched, summary.get("raw_records", []), strict=True):
        raw["failure_category"] = failure_category(item)
    summary["group"] = group_key
    return summary, enriched


def _build_focus_samples(group_results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for sid in FOCUS_SAMPLE_IDS:
        payload[sid] = {}
        for group_key, enriched in group_results.items():
            for item in enriched:
                if item.get("id") != sid:
                    continue
                raw = item.get("raw_record") or {}
                gen_debug = (raw.get("debug") or {}).get("generation_v2") or {}
                na = gen_debug.get("neighbor_audit") or {}
                payload[sid][group_key] = {
                    "id": sid,
                    "question": item.get("question"),
                    "answer_mode": raw.get("answer_mode"),
                    "citation_count": raw.get("citation_count"),
                    "support_pack_count": gen_debug.get("support_pack_count"),
                    "doc_hit": raw.get("doc_hit"),
                    "section_hit": raw.get("section_hit"),
                    "failure_category": failure_category(item),
                    "answer_preview": raw.get("answer_preview"),
                    "qwen_synthesis": gen_debug.get("qwen_synthesis") or {},
                    "comparison_coverage": gen_debug.get("comparison_coverage") or {},
                    "existence_guardrail": gen_debug.get("existence_guardrail") or {},
                    "neighbor_audit": {
                        "enabled": na.get("enabled", False),
                        "candidate_count": na.get("candidate_count", 0),
                        "dry_run_promoted_count": na.get("dry_run_promoted_count", 0),
                    },
                }
                break
    return payload


def _build_comparison_summary(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for gk, s in summaries.items():
        rows[gk] = {
            "label": s.get("label"),
            "route_match_rate": s.get("route_match_rate"),
            "doc_id_hit_rate": s.get("doc_id_hit_rate"),
            "section_hit_rate": s.get("section_hit_rate"),
            "answer_mode_distribution": s.get("answer_mode_distribution"),
            "citation_count_distribution": s.get("citation_count_distribution"),
            "qwen_used_count": s.get("qwen_used_count"),
            "qwen_fallback_count": s.get("qwen_fallback_count"),
            "failure_category_distribution": s.get("failure_category_distribution"),
        }
    return {"dataset": str(DATASET_PATH.relative_to(ROOT)), "groups": rows}


def _build_comparison_md(comp: dict[str, Any]) -> str:
    lines = [
        "# Generation V2 Baseline Matrix Comparison",
        "",
        f"Dataset: `{comp['dataset']}`",
        "",
        "| group | route_match | doc_id_hit | section_hit | answer_mode | qwen_used | qwen_fallback |",
        "|-------|-------------|------------|-------------|-------------|-----------|---------------|",
    ]
    for gk, row in comp["groups"].items():
        am = json.dumps(row.get("answer_mode_distribution") or {}, ensure_ascii=False)
        lines.append(
            f"| `{gk}` | `{row['route_match_rate']}` | `{row['doc_id_hit_rate']}` "
            f"| `{row['section_hit_rate']}` | `{am}` "
            f"| `{row['qwen_used_count']}` | `{row['qwen_fallback_count']}` |"
        )
    lines.append("")
    return "\n".join(lines)


# Module-level dataset cache (populated in main)
_records: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run generation v2 ablation baseline matrix against smoke20.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=list(ALL_GROUPS.keys()),
        default=list(ALL_GROUPS.keys()),
        metavar="GROUP",
        help=f"Which groups to run (default: all). Choices: {', '.join(ALL_GROUPS)}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print group config and exit without running any inference.",
    )
    return parser.parse_args()


def main() -> int:
    global _records

    args = _parse_args()

    if args.dry_run:
        print("=== dry-run: group config ===")
        for gk in args.groups:
            label, build_fn = ALL_GROUPS[gk]
            s = build_fn()
            print(json.dumps({
                "group": gk,
                "label": label,
                "generation_version": s.generation.version,
                "v2_use_qwen_synthesis": s.generation.v2_use_qwen_synthesis,
                "v2_enable_comparison_coverage": s.generation.v2_enable_comparison_coverage,
                "v2_enable_neighbor_audit": s.generation.v2_enable_neighbor_audit,
                "v2_enable_neighbor_promotion": s.generation.v2_enable_neighbor_promotion,
            }, indent=2, ensure_ascii=False))
        return 0

    _records = load_records(str(DATASET_PATH))
    run_root = (
        ROOT / "reports/evaluation/ad_hoc/generation_v2_baseline_matrix"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_root.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, Any]] = {}
    group_enriched: dict[str, list[dict[str, Any]]] = {}

    for gk in args.groups:
        label, build_fn = ALL_GROUPS[gk]
        print(f"[baseline_matrix] Running group: {gk} ...")
        summary, enriched = _run_group_with_settings(gk, label, build_fn)
        summaries[gk] = summary
        group_enriched[gk] = enriched
        (run_root / f"{gk}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_root / f"{gk}.md").write_text(
            build_markdown(summary), encoding="utf-8"
        )
        print(f"  -> route_match={summary['route_match_rate']} doc_hit={summary['doc_id_hit_rate']} section_hit={summary['section_hit_rate']}")

    comp = _build_comparison_summary(summaries)
    (run_root / "comparison_summary.json").write_text(
        json.dumps(comp, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_root / "comparison_summary.md").write_text(
        _build_comparison_md(comp), encoding="utf-8"
    )

    focus = _build_focus_samples(group_enriched)
    (run_root / "focus_samples.json").write_text(
        json.dumps(focus, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({
        "run_root": str(run_root.relative_to(ROOT)),
        "groups_run": list(summaries.keys()),
        "comparison_summary": comp,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
