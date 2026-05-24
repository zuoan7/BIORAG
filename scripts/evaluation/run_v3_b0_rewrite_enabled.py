from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from scripts.evaluation.run_v3_baseline_b0_b1 import (
    DATASET_PATH,
    REPORTS_ROOT,
    RESULTS_ROOT,
    VARIANTS,
    JudgeClient,
    build_skipped_probe,
    load_jsonl,
    render_failure_report,
    run_concurrency_probe,
    run_variant,
    summarize_variant,
    write_json,
    write_jsonl,
    write_markdown,
)


DEFAULT_BASELINE_SUMMARY = (
    RESULTS_ROOT
    / "v3_baseline_b0_b1_20260523_b0_b1_v3_fixed_metrics"
    / "b0_stable"
    / "summary.json"
)

RULE_METRICS_TO_COMPARE = (
    "doc_hit_at10_rate",
    "parent_chunk_hit_at10_rate",
    "exact_gold_chunk_hit_at10_rate",
    "support_parent_chunk_hit_rate",
    "support_child_evidence_hit_rate",
    "citation_parent_chunk_hit_rate",
    "citation_child_evidence_hit_rate",
    "route_match_rate",
)
JUDGE_METRICS_TO_COMPARE = (
    "answer_correctness_pass_rate",
    "critical_error_rate",
    "faithfulness_pass_rate",
    "citation_pass_rate",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v3 B0 with main-chain query rewrite enabled.")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--baseline-summary", default=str(DEFAULT_BASELINE_SUMMARY))
    parser.add_argument("--judge-concurrency", type=int, default=0)
    parser.add_argument("--judge-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--probe-concurrency", default="1,2,4,6,8")
    parser.add_argument("--probe-requests-per-level", type=int, default=8)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    args = parser.parse_args()

    load_dotenv(".env")
    dataset_path = Path(args.dataset)
    samples = load_jsonl(dataset_path)
    if args.limit > 0:
        samples = samples[: args.limit]

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = RESULTS_ROOT / f"v3_b0_rewrite_enabled_{run_id}"
    report_dir = REPORTS_ROOT / f"v3_b0_rewrite_enabled_{run_id}"
    result_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    cache_path = result_dir / "rewrite_cache.jsonl"
    cache_summary = write_rewrite_cache(samples, cache_path)
    spec = build_rewrite_b0_spec(cache_path)

    judge_required = not (args.skip_judge and args.skip_probe)
    judge = JudgeClient(
        timeout_seconds=args.judge_timeout_seconds,
        max_tokens=args.judge_max_tokens,
        required=judge_required,
    )
    if args.skip_probe:
        probe = build_skipped_probe(judge=judge, selected_concurrency=args.judge_concurrency or 1)
    else:
        levels = [int(v) for v in args.probe_concurrency.split(",") if v.strip()]
        probe = run_concurrency_probe(
            judge=judge,
            samples=samples,
            levels=levels,
            requests_per_level=args.probe_requests_per_level,
        )
    write_json(result_dir / "concurrency_probe.json", probe)

    selected_concurrency = args.judge_concurrency or int(probe["selected_concurrency"])
    variant_dir = result_dir / "b0_rewrite_enabled"
    variant_dir.mkdir(parents=True, exist_ok=True)
    rows = run_variant(
        variant_key="b0_rewrite_enabled",
        spec=spec,
        samples=samples,
        judge=judge,
        judge_concurrency=selected_concurrency,
        skip_judge=args.skip_judge,
        output_dir=variant_dir,
    )
    summary = summarize_variant(rows)
    summary["variant_key"] = "b0_rewrite_enabled"
    summary["variant_name"] = spec["display_name"]
    summary["variant_description"] = spec["description"]
    summary["config_overrides"] = spec["overrides"]
    write_json(variant_dir / "summary.json", summary)
    write_markdown(report_dir / "b0_rewrite_enabled_failures.md", render_failure_report(spec, rows))

    baseline_summary_path = Path(args.baseline_summary)
    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    comparison = compare_to_baseline(baseline_summary, summary)
    run_config = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "sample_count": len(samples),
        "baseline_summary": str(baseline_summary_path),
        "rewrite_cache": str(cache_path),
        "rewrite_cache_summary": cache_summary,
        "judge_model": judge.model,
        "selected_judge_concurrency": selected_concurrency,
        "skip_judge": args.skip_judge,
        "skip_probe": args.skip_probe,
        "variant": {
            "display_name": spec["display_name"],
            "description": spec["description"],
            "overrides": spec["overrides"],
        },
    }
    write_json(result_dir / "run_config.json", run_config)
    write_json(result_dir / "comparison_vs_b0_baseline.json", comparison)
    write_markdown(report_dir / "report.md", render_report(run_config, comparison, summary, baseline_summary))
    print(json.dumps({"output_dir": str(result_dir), "report_dir": str(report_dir)}, ensure_ascii=False))


def build_rewrite_b0_spec(cache_path: Path) -> dict[str, Any]:
    spec = copy.deepcopy(VARIANTS["b0_stable"])
    spec["display_name"] = "B0 Rewrite Enabled"
    spec["description"] = (
        "B0 stable with main-chain query_rewrite.mode=enabled using frozen "
        "English mirror cache; parent expansion and LLM synthesis remain disabled."
    )
    spec["overrides"]["query_rewrite.mode"] = "enabled"
    spec["overrides"]["query_rewrite.eval_rewrite_cache_path"] = str(cache_path)
    spec["overrides"]["query_rewrite.eval_rewrite_require_cache"] = True
    spec["overrides"]["query_rewrite.eval_rewrite_fail_fast_on_missing"] = False
    spec["overrides"]["query_rewrite.require_llm_for_eval"] = False
    return spec


def write_rewrite_cache(samples: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    rows = []
    source_counts: dict[str, int] = {}
    for sample in samples:
        original = str(sample.get("question") or "")
        rewritten = str(((sample.get("metadata") or {}).get("original_generated_question") or "")).strip()
        source = "metadata.original_generated_question"
        if not rewritten:
            rewritten = original
            source = "identity_fallback"
        rows.append(
            {
                "sample_id": sample.get("sample_id"),
                "original_query": original,
                "rewritten_query": rewritten,
                "source": source,
            }
        )
        source_counts[source] = source_counts.get(source, 0) + 1
    write_jsonl(path, rows)
    return {
        "entry_count": len(rows),
        "source_counts": source_counts,
        "identity_fallback_sample_ids": [
            str(row.get("sample_id") or "")
            for row in rows
            if row.get("source") == "identity_fallback"
        ],
    }


def compare_to_baseline(baseline: dict[str, Any], rewrite: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule_metrics": compare_metric_group(
            baseline.get("rule_metrics") or {},
            rewrite.get("rule_metrics") or {},
            RULE_METRICS_TO_COMPARE,
        ),
        "judge_metrics": compare_metric_group(
            baseline.get("judge_metrics") or {},
            rewrite.get("judge_metrics") or {},
            JUDGE_METRICS_TO_COMPARE,
        ),
        "latency": compare_metric_group(
            baseline.get("latency") or {},
            rewrite.get("latency") or {},
            ("rag_latency_avg_seconds", "rag_latency_p95_seconds"),
        ),
    }


def compare_metric_group(baseline: dict[str, Any], rewrite: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    result = {}
    for key in keys:
        before = baseline.get(key)
        after = rewrite.get(key)
        result[key] = {
            "baseline": before,
            "rewrite_enabled": after,
            "delta": round(float(after) - float(before), 6)
            if before is not None and after is not None
            else None,
        }
    return result


def render_report(
    run_config: dict[str, Any],
    comparison: dict[str, Any],
    rewrite_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
) -> str:
    lines = [
        "# B0 Rewrite Enabled 对比报告",
        "",
        f"- run_id: `{run_config['run_id']}`",
        f"- dataset: `{run_config['dataset_path']}`",
        f"- baseline_summary: `{run_config['baseline_summary']}`",
        f"- rewrite_cache: `{run_config['rewrite_cache']}`",
        f"- sample_count: {run_config['sample_count']}",
        f"- skip_judge: {run_config['skip_judge']}",
        f"- cache source counts: {json.dumps(run_config['rewrite_cache_summary']['source_counts'], ensure_ascii=False, sort_keys=True)}",
        "",
        "## Rule Metrics",
        "",
        "| metric | B0 baseline | B0 rewrite | delta |",
        "|---|---:|---:|---:|",
    ]
    for key, item in comparison["rule_metrics"].items():
        lines.append(f"| `{key}` | {fmt(item['baseline'])} | {fmt(item['rewrite_enabled'])} | {fmt_delta(item['delta'])} |")

    lines.extend(["", "## Judge Metrics", ""])
    lines.append("| metric | B0 baseline | B0 rewrite | delta |")
    lines.append("|---|---:|---:|---:|")
    for key, item in comparison["judge_metrics"].items():
        lines.append(f"| `{key}` | {fmt(item['baseline'])} | {fmt(item['rewrite_enabled'])} | {fmt_delta(item['delta'])} |")

    lines.extend(["", "## Latency", ""])
    lines.append("| metric | B0 baseline | B0 rewrite | delta |")
    lines.append("|---|---:|---:|---:|")
    for key, item in comparison["latency"].items():
        lines.append(f"| `{key}` | {fmt(item['baseline'])} | {fmt(item['rewrite_enabled'])} | {fmt_delta(item['delta'])} |")

    lines.extend(["", "## Notes", ""])
    lines.append("- Rewrite uses the main-chain `query_rewrite.mode=enabled` path with a frozen cache.")
    lines.append("- `metadata.original_generated_question` is used as the English mirror where available; missing entries use identity fallback.")
    lines.append("- Parent expansion, query LLM synthesis, table preview merge, and original-CN fallback remain disabled as in B0.")
    if run_config["skip_judge"]:
        lines.append("- Judge was skipped for this run; judge metric deltas are intentionally N/A.")
    lines.append("")
    lines.append("## Summary JSON Pointers")
    lines.append("")
    lines.append(f"- baseline sample_count: {baseline_summary.get('sample_count')}")
    lines.append(f"- rewrite sample_count: {rewrite_summary.get('sample_count')}")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def fmt_delta(value: Any) -> str:
    if value is None:
        return "N/A"
    numeric = float(value)
    prefix = "+" if numeric > 0 else ""
    return f"{prefix}{numeric:.6g}"


if __name__ == "__main__":
    main()
