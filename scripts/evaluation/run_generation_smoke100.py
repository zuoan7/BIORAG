#!/usr/bin/env python3
"""
Smoke100: old_baseline vs v2_stable comparison on enterprise_ragas_eval_v1 (100 samples).

Purpose: validate whether v2 stable profile can fully replace the old pipeline.

Pass condition (all must hold):
  - v2_stable route_match_rate  >= old_baseline route_match_rate  - 0.03
  - v2_stable doc_id_hit_rate   >= old_baseline doc_id_hit_rate   - 0.03
  - v2_stable section_hit_rate  >= old_baseline section_hit_rate  - 0.05

Usage
-----
  python scripts/evaluation/run_generation_smoke100.py
  python scripts/evaluation/run_generation_smoke100.py --dry-run
"""
from __future__ import annotations

import json
import sys
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

DATASET_PATH = ROOT / "data/eval/datasets/enterprise_ragas_eval_v1.json"

ROUTE_MARGIN = 0.03
DOC_MARGIN = 0.03
SECTION_MARGIN = 0.05


def _build_old_baseline() -> Settings:
    s = Settings.from_env()
    s.audit.enabled = False
    s.generation.version = "old"
    return s


def _build_v2_stable() -> Settings:
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


GROUPS: dict[str, tuple[str, Callable[[], Settings]]] = {
    "old_baseline": ("smoke100: old pipeline (legacy)", _build_old_baseline),
    "v2_stable":    ("smoke100: v2 stable profile (no Qwen)", _build_v2_stable),
}

_records: list[dict[str, Any]] = []


def _run_group(group_key: str, label: str, build_fn: Callable[[], Settings]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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


def _verdict(old: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "route_match": {
            "old": old["route_match_rate"],
            "v2":  v2["route_match_rate"],
            "threshold": round(float(old["route_match_rate"] or 0) - ROUTE_MARGIN, 4),
            "pass": float(v2["route_match_rate"] or 0) >= float(old["route_match_rate"] or 0) - ROUTE_MARGIN,
        },
        "doc_id_hit": {
            "old": old["doc_id_hit_rate"],
            "v2":  v2["doc_id_hit_rate"],
            "threshold": round(float(old["doc_id_hit_rate"] or 0) - DOC_MARGIN, 4),
            "pass": float(v2["doc_id_hit_rate"] or 0) >= float(old["doc_id_hit_rate"] or 0) - DOC_MARGIN,
        },
        "section_hit": {
            "old": old["section_hit_rate"],
            "v2":  v2["section_hit_rate"],
            "threshold": round(float(old["section_hit_rate"] or 0) - SECTION_MARGIN, 4),
            "pass": float(v2["section_hit_rate"] or 0) >= float(old["section_hit_rate"] or 0) - SECTION_MARGIN,
        },
    }
    overall = all(c["pass"] for c in checks.values())
    return {"checks": checks, "overall_pass": overall}


def _comparison_md(summaries: dict[str, dict[str, Any]], verdict: dict[str, Any]) -> str:
    old = summaries["old_baseline"]
    v2  = summaries["v2_stable"]
    checks = verdict["checks"]
    lines = [
        "# Smoke100: old_baseline vs v2_stable",
        "",
        f"Dataset: `{DATASET_PATH.relative_to(ROOT)}`  (N=100)",
        "",
        "## Metrics",
        "",
        "| metric | old_baseline | v2_stable | threshold | pass |",
        "|--------|-------------|-----------|-----------|------|",
    ]
    for key, c in checks.items():
        icon = "✅" if c["pass"] else "❌"
        lines.append(f"| {key} | `{c['old']}` | `{c['v2']}` | `>={c['threshold']}` | {icon} |")
    lines += [
        "",
        f"**Overall: {'PASS — v2_stable can replace old' if verdict['overall_pass'] else 'FAIL — v2_stable does NOT meet threshold'}**",
        "",
        "## Answer Mode Distribution",
        "",
        f"- old: `{json.dumps(old['answer_mode_distribution'], ensure_ascii=False)}`",
        f"- v2:  `{json.dumps(v2['answer_mode_distribution'], ensure_ascii=False)}`",
        "",
        "## Failure Category Distribution",
        "",
        f"- old: `{json.dumps(old['failure_category_distribution'], ensure_ascii=False)}`",
        f"- v2:  `{json.dumps(v2['failure_category_distribution'], ensure_ascii=False)}`",
        "",
    ]

    v2_ledger = (v2 or {}).get("retrieval_ledger") or {}
    old_ledger = (old or {}).get("retrieval_ledger") or {}
    if v2_ledger:
        lines += [
            "## Retrieval Ledger (v2 diagnostic)",
            "",
            "Pipeline: candidate → support_pack → citation",
            "",
            "| Stage | v2_doc_hit | v2_section_hit |",
            "|---|---|---|",
            f"| candidate | `{v2_ledger.get('candidate_doc_hit_rate')}` | `{v2_ledger.get('candidate_section_hit_rate')}` |",
            f"| support_pack | `{v2_ledger.get('support_doc_hit_rate')}` | `{v2_ledger.get('support_section_hit_rate')}` |",
            f"| citation | `{v2_ledger.get('citation_doc_hit_rate')}` | `{v2_ledger.get('citation_section_hit_rate')}` |",
            "",
            f"v2 doc_status: `{json.dumps(v2_ledger.get('doc_status_distribution'), ensure_ascii=False)}`",
            f"v2 section_status: `{json.dumps(v2_ledger.get('section_status_distribution'), ensure_ascii=False)}`",
            f"v2 section_label_issue_count: `{v2_ledger.get('section_label_issue_count')}`",
        ]
        if v2_ledger.get("section_label_issue_ids"):
            lines.append(f"v2 section_label_issue_ids: {', '.join(f'`{i}`' for i in v2_ledger['section_label_issue_ids'])}")
        lines.append("")
    if old_ledger:
        lines += [
            "## Retrieval Ledger (old diagnostic)",
            "",
            f"old doc_status: `{json.dumps(old_ledger.get('doc_status_distribution'), ensure_ascii=False)}`",
            f"old section_status: `{json.dumps(old_ledger.get('section_status_distribution'), ensure_ascii=False)}`",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    global _records

    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("=== dry-run: group config ===")
        for gk, (label, build_fn) in GROUPS.items():
            s = build_fn()
            print(json.dumps({
                "group": gk,
                "label": label,
                "generation_version": s.generation.version,
                "v2_use_qwen_synthesis": s.generation.v2_use_qwen_synthesis,
                "v2_enable_comparison_coverage": s.generation.v2_enable_comparison_coverage,
                "v2_enable_neighbor_promotion": s.generation.v2_enable_neighbor_promotion,
            }, indent=2, ensure_ascii=False))
        return 0

    _records = load_records(str(DATASET_PATH))
    run_root = (
        ROOT / "reports/evaluation/ad_hoc/generation_smoke100"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_root.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, Any]] = {}
    for gk, (label, build_fn) in GROUPS.items():
        print(f"[smoke100] Running group: {gk} ...")
        summary, _ = _run_group(gk, label, build_fn)
        summaries[gk] = summary
        (run_root / f"{gk}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_root / f"{gk}.md").write_text(build_markdown(summary), encoding="utf-8")
        print(f"  -> route={summary['route_match_rate']} doc={summary['doc_id_hit_rate']} section={summary['section_hit_rate']} mode={summary['answer_mode_distribution']}")

    verdict = _verdict(summaries["old_baseline"], summaries["v2_stable"])
    (run_root / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_root / "comparison.md").write_text(_comparison_md(summaries, verdict), encoding="utf-8")

    print("\n" + "=" * 60)
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print("=" * 60)
    print(f"Reports: {run_root.relative_to(ROOT)}")
    return 0 if verdict["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
