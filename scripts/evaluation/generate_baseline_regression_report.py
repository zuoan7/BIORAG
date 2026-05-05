#!/usr/bin/env python3
"""Generate a warning-only smoke100 regression report against Phase 9 baseline."""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE9_BASELINE_DIR = ROOT / "results/ragas/smoke100_20260504_214135"
PHASE9_NOISE_LEDGER = PHASE9_BASELINE_DIR / "phase10b_evaluation_noise_ledger.csv"

PHASE9_BASELINE = {
    "name": "Phase 9 accepted baseline",
    "result_dir": "results/ragas/smoke100_20260504_214135/",
    "faithfulness": 0.6886,
    "answer_relevancy": 0.3185,
    "raw_p0_count": 10,
    "rule_review_candidate_count": 8,
    "noise_adjusted_p0_count": 3,
    "calibrated_p0_count": 8,
    "false_refusal": 0,
    "new_hallucination": 0,
    "qwen_citation_loss_count": 0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate warning-only baseline regression report for smoke100 outputs."
    )
    p.add_argument("--result-dir", required=True, help="smoke100 result directory")
    p.add_argument("--baseline-dir", default=str(PHASE9_BASELINE_DIR),
                   help="Phase 9 accepted baseline result directory")
    p.add_argument("--noise-ledger", default=str(PHASE9_NOISE_LEDGER),
                   help="Phase 10B evaluation noise ledger CSV")
    p.add_argument("--output-dir", default="",
                   help="Output directory for report files (default: --result-dir)")
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_noise_ledger(path: Path) -> dict[str, dict[str, str]]:
    rows = load_csv(path)
    noise: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id:
            continue
        if not _as_bool(row.get("should_count_as_p0", "true")):
            noise[sample_id] = row
    return noise


def _score(summary: dict[str, Any], metric: str) -> float | None:
    value = (summary.get("global_averages") or {}).get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def _qwen_enabled(env_overrides: dict[str, str]) -> bool:
    return str(env_overrides.get("GENERATION_V2_USE_QWEN_SYNTHESIS", "")).lower() == "true"


def audit_qwen_citation_loss(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit citation loss using final response invariants available in RAGAS JSONL.

    Qwen is allowed to rewrite text but must not produce a substantive answer with
    zero bound citations. The evaluation dataset no longer stores generation_v2
    debug, so this audit uses the final `answer_mode` and `citation_count`.
    """
    loss_ids: list[str] = []
    marker_loss_ids: list[str] = []
    citation_marker = re.compile(r"\[\d+\]")
    for item in scores:
        sid = str(item.get("sample_id") or "")
        mode = str(item.get("answer_mode") or "")
        citation_count = item.get("citation_count") or 0
        try:
            citation_count = int(citation_count)
        except (TypeError, ValueError):
            citation_count = 0
        answer = str(item.get("answer") or "")
        if mode not in {"refusal", "empty", "error"} and citation_count == 0:
            loss_ids.append(sid)
        if citation_count > 0 and answer and not citation_marker.search(answer):
            marker_loss_ids.append(sid)
    return {
        "qwen_citation_loss_count": len(loss_ids),
        "qwen_citation_loss_sample_ids": loss_ids,
        "citation_marker_loss_count": len(marker_loss_ids),
        "citation_marker_loss_sample_ids": marker_loss_ids,
        "method": "substantive_non_refusal_with_zero_final_citations",
    }


def collect_metrics(result_dir: Path, noise_ledger: dict[str, dict[str, str]]) -> dict[str, Any]:
    ragas_summary = load_json(result_dir / "ragas_summary.json")
    calibration_summary = load_json(result_dir / "calibration_summary.json")
    candidates = load_csv(result_dir / "human_review_candidates_calibrated.csv")
    scores = load_jsonl(result_dir / "ragas_scores.jsonl")
    manifest = load_json(result_dir / "smoke100_pipeline_manifest.json")
    risk_check = load_json(result_dir / "phase10a_phase9_risk_check.json")

    p0_ids = list(calibration_summary.get("calibrated_p0_sample_ids") or [])
    if not p0_ids and candidates:
        p0_ids = [
            row.get("sample_id", "")
            for row in candidates
            if row.get("calibrated_priority") == "P0"
        ]

    excluded_noise = []
    for sample_id in p0_ids:
        if sample_id in noise_ledger:
            row = noise_ledger[sample_id]
            excluded_noise.append({
                "sample_id": sample_id,
                "issue_type": row.get("issue_type", ""),
                "reason": row.get("reason", ""),
            })

    adjusted_p0_ids = [sid for sid in p0_ids if sid not in noise_ledger]

    false_refusal = 0
    for row in candidates:
        if row.get("calibrated_priority") == "P0" and "false_refusal" in row.get("calibrated_issue_type", ""):
            false_refusal += 1

    human_hallucination_yes = [
        row.get("sample_id", "")
        for row in candidates
        if (row.get("human_hallucination") or "").strip().lower() in {"yes", "true", "1"}
    ]
    if "qwen_added_unsupported_facts" in risk_check:
        new_hallucination = int(bool(risk_check.get("qwen_added_unsupported_facts")))
    else:
        new_hallucination = len(human_hallucination_yes)

    env_overrides = manifest.get("preset_env_overrides") or manifest.get("stable_env_overrides") or {}
    qwen_enabled = _qwen_enabled(env_overrides)
    qwen_audit = audit_qwen_citation_loss(scores) if scores else {}
    if "qwen_lost_citation" in risk_check:
        qwen_citation_loss = int(bool(risk_check.get("qwen_lost_citation")))
        qwen_citation_loss_note = "Read from phase10a_phase9_risk_check.json."
        qwen_citation_loss_comparable = True
    elif not qwen_enabled:
        qwen_citation_loss = 0
        qwen_citation_loss_note = "Qwen synthesis disabled for this run."
        qwen_citation_loss_comparable = True
    elif qwen_audit:
        qwen_citation_loss = int(qwen_audit["qwen_citation_loss_count"])
        qwen_citation_loss_note = (
            "Auto-audited from final answer_mode/citation_count: "
            "substantive non-refusal with zero final citations."
        )
        qwen_citation_loss_comparable = True
    else:
        qwen_citation_loss = None
        qwen_citation_loss_note = "Qwen synthesis enabled but citation loss audit could not run."
        qwen_citation_loss_comparable = False

    comparability = manifest.get("comparability") or {}
    if not qwen_citation_loss_comparable:
        comparability = dict(comparability)
        comparability["comparable_to_phase9"] = False
        reasons = list(comparability.get("non_comparable_reasons") or [])
        reasons.append("qwen citation loss audit unavailable")
        comparability["non_comparable_reasons"] = reasons

    return {
        "result_dir": str(result_dir),
        "sample_count": ragas_summary.get("sample_count") or len(scores),
        "faithfulness": _score(ragas_summary, "faithfulness"),
        "answer_relevancy": _score(ragas_summary, "answer_relevancy"),
        "raw_p0_count": calibration_summary.get("original_p0_count"),
        "rule_review_candidate_count": calibration_summary.get("calibrated_p0_count"),
        "rule_review_candidate_sample_ids": p0_ids,
        "noise_adjusted_p0_count": len(adjusted_p0_ids),
        "noise_adjusted_p0_sample_ids": adjusted_p0_ids,
        # Backward-compatible aliases for older Phase 11B reports.
        "rule_calibrated_p0_count": calibration_summary.get("calibrated_p0_count"),
        "calibrated_p0_count": len(adjusted_p0_ids),
        "calibrated_p0_sample_ids": adjusted_p0_ids,
        "rule_calibrated_p0_sample_ids": p0_ids,
        "noise_excluded_count": len(excluded_noise),
        "noise_excluded_samples": excluded_noise,
        "false_refusal": false_refusal,
        "new_hallucination": new_hallucination,
        "new_hallucination_sample_ids": human_hallucination_yes,
        "qwen_citation_loss_count": qwen_citation_loss,
        "qwen_citation_loss": qwen_citation_loss,
        "qwen_citation_loss_note": qwen_citation_loss_note,
        "qwen_citation_loss_comparable": qwen_citation_loss_comparable,
        "qwen_citation_loss_audit": qwen_audit,
        "run_config": manifest.get("config") or {},
        "preset_env_overrides": manifest.get("preset_env_overrides") or env_overrides,
        "stable_env_overrides": env_overrides,
        "dataset": manifest.get("dataset") or {},
        "comparability": comparability,
    }


def compare(current: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(metric: str, current_value: Any, baseline_value: Any,
            warning: bool, note: str) -> None:
        delta = None
        if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
            delta = round(float(current_value) - float(baseline_value), 4)
        checks.append({
            "metric": metric,
            "current": current_value,
            "baseline": baseline_value,
            "delta": delta,
            "status": "WARNING" if warning else "OK",
            "note": note,
        })

    faith = current.get("faithfulness")
    add(
        "faithfulness",
        faith,
        PHASE9_BASELINE["faithfulness"],
        isinstance(faith, (int, float)) and faith < PHASE9_BASELINE["faithfulness"] - 0.03,
        "Warning if drop > 0.03.",
    )

    arel = current.get("answer_relevancy")
    add(
        "answer_relevancy",
        arel,
        PHASE9_BASELINE["answer_relevancy"],
        isinstance(arel, (int, float)) and arel < PHASE9_BASELINE["answer_relevancy"] - 0.03,
        "Warning if drop > 0.03.",
    )

    add(
        "raw_p0_count",
        current.get("raw_p0_count"),
        PHASE9_BASELINE["raw_p0_count"],
        isinstance(current.get("raw_p0_count"), int)
        and current["raw_p0_count"] > PHASE9_BASELINE["raw_p0_count"],
        "Pre-noise candidate P0 count.",
    )

    add(
        "rule_review_candidate_count",
        current.get("rule_review_candidate_count"),
        PHASE9_BASELINE["rule_review_candidate_count"],
        isinstance(current.get("rule_review_candidate_count"), int)
        and current["rule_review_candidate_count"] > PHASE9_BASELINE["rule_review_candidate_count"],
        "Rule-generated review P0 candidates; not guaranteed to be a subset of raw P0.",
    )

    add(
        "noise_adjusted_p0_count",
        current.get("noise_adjusted_p0_count"),
        PHASE9_BASELINE["noise_adjusted_p0_count"],
        isinstance(current.get("noise_adjusted_p0_count"), int)
        and current["noise_adjusted_p0_count"] > PHASE9_BASELINE["noise_adjusted_p0_count"],
        "After excluding Phase 10B evaluation noise ledger.",
    )

    add(
        "false_refusal",
        current.get("false_refusal"),
        PHASE9_BASELINE["false_refusal"],
        isinstance(current.get("false_refusal"), int) and current["false_refusal"] > 0,
        "Warning if any false refusal appears.",
    )

    add(
        "new_hallucination",
        current.get("new_hallucination"),
        PHASE9_BASELINE["new_hallucination"],
        isinstance(current.get("new_hallucination"), int) and current["new_hallucination"] > 0,
        "Automated report only counts reviewed hallucination flags if present.",
    )

    qwen_loss = current.get("qwen_citation_loss_count")
    add(
        "qwen_citation_loss_count",
        qwen_loss,
        PHASE9_BASELINE["qwen_citation_loss_count"],
        (qwen_loss is None) or (isinstance(qwen_loss, int) and qwen_loss > 0),
        current.get("qwen_citation_loss_note", ""),
    )

    return checks


def build_markdown(current: dict[str, Any], checks: list[dict[str, Any]],
                   baseline_dir: Path, noise_ledger_path: Path) -> str:
    lines = [
        "# Smoke100 Baseline Regression Report",
        "",
        f"**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Current result dir**: `{current['result_dir']}`",
        f"**Baseline**: {PHASE9_BASELINE['name']} (`{PHASE9_BASELINE['result_dir']}`)",
        f"**Baseline dir read**: `{baseline_dir}`",
        f"**Noise ledger**: `{noise_ledger_path}`",
        "",
        "> First version: warning-only. This report does not hard fail the pipeline.",
        "",
    ]
    comparability = current.get("comparability") or {}
    if comparability:
        comparable = bool(comparability.get("comparable_to_phase9"))
        lines += [
            "## Comparability",
            "",
            f"- Comparable to Phase 9 accepted baseline: `{comparable}`",
        ]
        reasons = comparability.get("non_comparable_reasons") or []
        if reasons:
            lines.append("- Non-comparable reasons:")
            for reason in reasons:
                lines.append(f"  - {reason}")
        lines.append("")

    lines += [
        "## Metrics",
        "",
        "| Metric | Current | Phase 9 baseline | Delta | Status | Note |",
        "|--------|---------|------------------|-------|--------|------|",
    ]
    for item in checks:
        lines.append(
            f"| {item['metric']} | `{item['current']}` | `{item['baseline']}` | "
            f"`{item['delta']}` | {item['status']} | {item['note']} |"
        )

    lines += [
        "",
        "## Noise Ledger Calibration",
        "",
        f"- Raw P0 count: `{current.get('raw_p0_count')}`",
        f"- Rule review candidate count: `{current.get('rule_review_candidate_count')}`",
        f"- Noise-adjusted P0 count: `{current.get('noise_adjusted_p0_count')}`",
        f"- Excluded noise count: `{current.get('noise_excluded_count')}`",
        "",
    ]

    excluded = current.get("noise_excluded_samples") or []
    if excluded:
        lines += ["| Sample | Noise type | Reason |", "|--------|------------|--------|"]
        for item in excluded:
            reason = str(item.get("reason", "")).replace("|", "\\|")
            lines.append(f"| `{item.get('sample_id')}` | `{item.get('issue_type')}` | {reason} |")
    else:
        lines.append("No calibrated P0 samples matched the configured noise ledger.")

    lines += [
        "",
        "## Run Configuration",
        "",
        "```json",
        json.dumps({
            "run_config": current.get("run_config") or {},
            "preset_env_overrides": current.get("preset_env_overrides") or {},
            "dataset": current.get("dataset") or {},
            "comparability": current.get("comparability") or {},
            "qwen_citation_loss_audit": current.get("qwen_citation_loss_audit") or {},
        }, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Notes",
        "",
        "- `raw_p0_count` is the original pre-calibration candidate count.",
        "- `rule_review_candidate_count` is the clearer name for the old `rule_calibrated_p0_count`; it can include candidates promoted from non-raw P0 triggers.",
        "- `noise_adjusted_p0_count` excludes samples marked `should_count_as_p0=false` in the Phase 10B noise ledger.",
        "- Qwen citation loss is audited as substantive non-refusal answers with zero final citations when Qwen synthesis is enabled.",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    result_dir = Path(args.result_dir)
    baseline_dir = Path(args.baseline_dir)
    noise_ledger_path = Path(args.noise_ledger)
    output_dir = Path(args.output_dir) if args.output_dir else result_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    noise = load_noise_ledger(noise_ledger_path)
    current = collect_metrics(result_dir, noise)
    checks = compare(current)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "status": (
            "warning_only_comparable"
            if (current.get("comparability") or {}).get("comparable_to_phase9")
            else "warning_only_non_comparable"
        ),
        "baseline": PHASE9_BASELINE,
        "current": current,
        "checks": checks,
    }

    json_path = output_dir / "baseline_regression_report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = output_dir / "baseline_regression_report.md"
    md_path.write_text(
        build_markdown(current, checks, baseline_dir, noise_ledger_path),
        encoding="utf-8",
    )

    warning_count = sum(1 for item in checks if item["status"] == "WARNING")
    print(f"[baseline_report] Wrote {json_path}")
    print(f"[baseline_report] Wrote {md_path}")
    print(f"[baseline_report] warnings={warning_count} hard_fail=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
