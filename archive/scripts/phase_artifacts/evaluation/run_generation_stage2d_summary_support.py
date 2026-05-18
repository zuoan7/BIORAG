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


def _latest_stage2c3_summary_path() -> Path:
    base_dir = ROOT / "reports/evaluation/ad_hoc/generation_v2_stage2c3_validator_polish"
    candidates = sorted(
        path / "v2_stage2c3_current.json"
        for path in base_dir.iterdir()
        if path.is_dir() and (path / "v2_stage2c3_current.json").exists()
    )
    if not candidates:
        raise FileNotFoundError("No Stage 2C.3 baseline report found.")
    return candidates[-1]


def _support_pack_distribution(summary: dict[str, Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in summary.get("raw_records", []):
        debug = ((record.get("debug") or {}).get("generation_v2") or {})
        counter[str(debug.get("support_pack_count", 0))] += 1
    return dict(sorted(counter.items(), key=lambda entry: int(entry[0])))


def _failure_distribution(summary: dict[str, Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in summary.get("raw_records", []):
        category = str(record.get("failure_category") or "ok")
        counter[category] += 1
    return dict(sorted(counter.items()))


def _focus_payload(current_summary: dict[str, Any], baseline_summary: dict[str, Any]) -> dict[str, Any]:
    by_id = {
        "stage2c3": {record.get("id"): record for record in baseline_summary.get("raw_records", [])},
        "stage2d": {record.get("id"): record for record in current_summary.get("raw_records", [])},
    }
    payload: dict[str, Any] = {}
    for sample_id in FOCUS_SAMPLE_IDS:
        payload[sample_id] = {}
        for label, records in by_id.items():
            record = records.get(sample_id)
            if not record:
                continue
            gen_debug = ((record.get("debug") or {}).get("generation_v2") or {})
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
                "comparison_coverage": gen_debug.get("comparison_coverage") or {},
                "qwen_synthesis": gen_debug.get("qwen_synthesis") or {},
                "support_pack": gen_debug.get("support_pack") or [],
            }
    return payload


def _diff_vs_stage2c3(baseline_summary: dict[str, Any], current_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_stage2c3_path": str(_latest_stage2c3_summary_path().relative_to(ROOT)),
        "metrics": {
            "route_match_rate": {
                "stage2c3": baseline_summary.get("route_match_rate"),
                "stage2d": current_summary.get("route_match_rate"),
            },
            "doc_id_hit_rate": {
                "stage2c3": baseline_summary.get("doc_id_hit_rate"),
                "stage2d": current_summary.get("doc_id_hit_rate"),
            },
            "section_hit_rate": {
                "stage2c3": baseline_summary.get("section_hit_rate"),
                "stage2d": current_summary.get("section_hit_rate"),
            },
            "answer_mode_distribution": {
                "stage2c3": baseline_summary.get("answer_mode_distribution"),
                "stage2d": current_summary.get("answer_mode_distribution"),
            },
            "citation_count_distribution": {
                "stage2c3": baseline_summary.get("citation_count_distribution"),
                "stage2d": current_summary.get("citation_count_distribution"),
            },
            "support_pack_count_distribution": {
                "stage2c3": _support_pack_distribution(baseline_summary),
                "stage2d": _support_pack_distribution(current_summary),
            },
            "failure_category_distribution": {
                "stage2c3": _failure_distribution(baseline_summary),
                "stage2d": _failure_distribution(current_summary),
            },
        },
    }


def main() -> int:
    dataset_path = ROOT / "data/eval/datasets/enterprise_ragas_smoke20.json"
    records = load_records(str(dataset_path))
    run_root = ROOT / "reports/evaluation/ad_hoc/generation_v2_stage2d_summary_support" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    current_summary, enriched = run_group("v2_stage2c_current", "v2 Stage 2D summary support", dataset_path, records)
    current_summary["group"] = "v2_stage2d_current"
    baseline_path = _latest_stage2c3_summary_path()
    baseline_summary = json.loads(baseline_path.read_text(encoding="utf-8"))

    for item, raw in zip(enriched, current_summary.get("raw_records", []), strict=True):
        raw["failure_category"] = failure_category(item)

    baseline_json = run_root / "v2_stage2c3_baseline.json"
    baseline_json.write_text(json.dumps(baseline_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    current_json = run_root / "v2_stage2d_current.json"
    current_json.write_text(json.dumps(current_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    current_md = run_root / "v2_stage2d_current.md"
    current_md.write_text(build_markdown(current_summary), encoding="utf-8")

    diff_json = run_root / "diff_vs_stage2c3.json"
    diff_json.write_text(json.dumps(_diff_vs_stage2c3(baseline_summary, current_summary), ensure_ascii=False, indent=2), encoding="utf-8")

    focus_json = run_root / "focus_samples.json"
    focus_json.write_text(json.dumps(_focus_payload(current_summary, baseline_summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_root": str(run_root.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
