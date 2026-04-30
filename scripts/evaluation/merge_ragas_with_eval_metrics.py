#!/usr/bin/env python3
"""
合并 RAGAS 分数与 smoke100 项目自定义指标，生成综合对比报告。

输入:
  - ragas_scores.jsonl (from run_ragas_smoke100.py)
  - smoke100 eval records / diagnostics ledger (from run_generation_smoke100.py or evaluate_ragas.py)

输出:
  - ragas_eval_joined.jsonl    merged per-sample records
  - ragas_eval_joined_summary.md  cross-analysis report

使用方式:
  python scripts/evaluation/merge_ragas_with_eval_metrics.py \\
    --ragas results/ragas/smoke100_<ts>/ragas_scores.jsonl \\
    --ledger reports/evaluation/ad_hoc/generation_smoke100/<ts>/eval_metric_diagnostics_ledger_v2_stable.json \\
    --output-dir results/ragas/<ts>/
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge RAGAS scores with project eval metrics")
    p.add_argument("--ragas", required=True, help="Path to ragas_scores.jsonl")
    p.add_argument("--ledger", default="", help="Path to diagnostics ledger JSON")
    p.add_argument("--output-dir", required=True, help="Output directory for merged files")
    return p.parse_args()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_json(path: str) -> dict[str, Any] | list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_merged(ragas_records: list[dict[str, Any]],
                 ledger: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Merge RAGAS scores with ledger retrieval diagnostics."""
    ledger_map: dict[str, dict[str, Any]] = {}
    if ledger and "ledger" in ledger:
        for item in ledger["ledger"]:
            sid = item.get("sample_id", "")
            if sid:
                ledger_map[sid] = item

    merged: list[dict[str, Any]] = []
    for rec in ragas_records:
        sid = rec.get("sample_id", "")
        entry = dict(rec)
        litem = ledger_map.get(sid, {})

        entry["ledger_doc_hit"] = bool(litem.get("expected_docs_in_citations"))
        entry["ledger_section_norm_hit"] = litem.get("normalized_section_hit")
        entry["ledger_answer_type"] = litem.get("answer_type", "")
        entry["ledger_section_failure"] = litem.get("section_failure_category", "")
        entry["ledger_doc_failure"] = litem.get("doc_failure_category", "")

        merged.append(entry)

    return merged


def find_cross_patterns(merged: list[dict[str, Any]],
                        metric_names: list[str]) -> dict[str, list[str]]:
    """Find interesting cross-patterns between RAGAS and project metrics."""
    patterns: dict[str, list[str]] = {
        "section_norm_hit_true__faithfulness_low": [],
        "section_norm_hit_false__faithfulness_high": [],
        "doc_id_hit_false__factual_correctness_high": [],
        "citation_count_zero__faithfulness_low": [],
        "citation_count_zero__factual_correctness_low": [],
    }

    for item in merged:
        sid = item.get("sample_id", "")
        scores = item.get("ragas_scores") or {}
        section_hit = item.get("section_norm_hit", item.get("ledger_section_norm_hit"))
        doc_hit = item.get("doc_id_hit", item.get("ledger_doc_hit"))
        cit_count = item.get("citation_count", 0)
        faith = scores.get("faithfulness")
        fc = scores.get("factual_correctness")

        # Pattern 1: section hit but low faithfulness
        if section_hit and isinstance(faith, (int, float)) and faith < 0.6:
            patterns["section_norm_hit_true__faithfulness_low"].append(sid)

        # Pattern 2: section miss but high faithfulness
        if section_hit is False and isinstance(faith, (int, float)) and faith > 0.8:
            patterns["section_norm_hit_false__faithfulness_high"].append(sid)

        # Pattern 3: doc miss but high factual correctness
        if doc_hit is False and isinstance(fc, (int, float)) and fc > 0.8:
            patterns["doc_id_hit_false__factual_correctness_high"].append(sid)

        # Pattern 4: zero citations + low faithfulness
        if cit_count == 0 and isinstance(faith, (int, float)) and faith < 0.5:
            patterns["citation_count_zero__faithfulness_low"].append(sid)

        # Pattern 5: zero citations + low factual correctness
        if cit_count == 0 and isinstance(fc, (int, float)) and fc < 0.5:
            patterns["citation_count_zero__factual_correctness_low"].append(sid)

    return patterns


def build_joined_summary_md(merged: list[dict[str, Any]],
                            patterns: dict[str, list[str]],
                            metric_names: list[str]) -> str:
    n = len(merged)
    lines = [
        "# RAGAS × Project Metrics — Joined Analysis",
        "",
        f"**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Samples**: {n}",
        "",
        "## Cross-Pattern Analysis",
        "",
        "### 1. section_norm_hit=true but faithfulness low",
        "",
        f"Count: {len(patterns['section_norm_hit_true__faithfulness_low'])}",
        "",
        "These samples had the expected section in citations, but the answer was not faithful to the evidence. "
        "Likely causes: answer synthesized claims beyond what the cited chunks support, or the model extrapolated.",
        "",
    ]
    if patterns["section_norm_hit_true__faithfulness_low"]:
        lines.append("Sample IDs: " + ", ".join(f"`{s}`" for s in patterns["section_norm_hit_true__faithfulness_low"][:20]))

    lines += [
        "",
        "### 2. section_norm_hit=false but faithfulness high",
        "",
        f"Count: {len(patterns['section_norm_hit_false__faithfulness_high'])}",
        "",
        "Section label missed (e.g. label mismatch), but answer was still faithful to retrieved context. "
        "Suggests the section label may need review, not the answer quality.",
        "",
    ]
    if patterns["section_norm_hit_false__faithfulness_high"]:
        lines.append("Sample IDs: " + ", ".join(f"`{s}`" for s in patterns["section_norm_hit_false__faithfulness_high"][:20]))

    lines += [
        "",
        "### 3. doc_id_hit=false but factual_correctness high",
        "",
        f"Count: {len(patterns['doc_id_hit_false__factual_correctness_high'])}",
        "",
        "Expected doc not in citations, but answer still matches reference facts. "
        "Possible: reference/expected_doc annotation needs review, or the model found the info in related documents.",
        "",
    ]
    if patterns["doc_id_hit_false__factual_correctness_high"]:
        lines.append("Sample IDs: " + ", ".join(f"`{s}`" for s in patterns["doc_id_hit_false__factual_correctness_high"][:20]))

    lines += [
        "",
        "### 4. citation_count=0 and faithfulness low",
        "",
        f"Count: {len(patterns['citation_count_zero__faithfulness_low'])}",
        "",
        "High-risk hallucination or incomplete refusal. These samples have no citations and low faithfulness — "
        "the model may be fabricating content or providing uninformative refusals.",
        "",
    ]
    if patterns["citation_count_zero__faithfulness_low"]:
        lines.append("Sample IDs: " + ", ".join(f"`{s}`" for s in patterns["citation_count_zero__faithfulness_low"][:20]))

    lines += [
        "",
        "### 5. citation_count=0 and factual_correctness low",
        "",
        f"Count: {len(patterns['citation_count_zero__factual_correctness_low'])}",
        "",
        "No citations and low factual correctness — failed on both retrieval and answer quality.",
        "",
    ]
    if patterns["citation_count_zero__factual_correctness_low"]:
        lines.append("Sample IDs: " + ", ".join(f"`{s}`" for s in patterns["citation_count_zero__factual_correctness_low"][:20]))

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ragas_records = load_jsonl(args.ragas)
    print(f"[merge] Loaded {len(ragas_records)} RAGAS records")

    ledger = None
    if args.ledger:
        ledger_data = load_json(args.ledger)
        if isinstance(ledger_data, dict):
            ledger = ledger_data
        print(f"[merge] Loaded ledger: group={ledger.get('group', 'unknown')}")

    merged = build_merged(ragas_records, ledger)

    # Detect metric names from first record with scores
    metric_names: list[str] = []
    for rec in merged:
        scores = rec.get("ragas_scores") or {}
        if scores:
            metric_names = list(scores.keys())
            break

    patterns = find_cross_patterns(merged, metric_names)

    # Write outputs
    joined_path = output_dir / "ragas_eval_joined.jsonl"
    with joined_path.open("w", encoding="utf-8") as fh:
        for item in merged:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[merge] Wrote {len(merged)} records → {joined_path}")

    md_path = output_dir / "ragas_eval_joined_summary.md"
    md_path.write_text(build_joined_summary_md(merged, patterns, metric_names), encoding="utf-8")
    print(f"[merge] Wrote joined summary → {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
