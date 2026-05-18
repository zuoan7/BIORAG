#!/usr/bin/env python3
"""Generate sample-level P0 reconciliation diff between two smoke100 runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data/eval/datasets/enterprise_ragas_smoke100.json"
DEFAULT_NOISE_LEDGER = (
    ROOT / "results/ragas/smoke100_20260504_214135/phase10b_evaluation_noise_ledger.csv"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Phase 9 vs current P0 diff table")
    p.add_argument("--phase9-dir", required=True)
    p.add_argument("--current-dir", required=True)
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--noise-ledger", default=str(DEFAULT_NOISE_LEDGER))
    p.add_argument("--output-dir", default="")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        for key in ("samples", "items", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"Unsupported dataset format: {path}")
    return {str(item.get("id") or item.get("sample_id")): item for item in data}


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None or value == "":
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if value is None or value == "":
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def candidate_map(run_dir: Path) -> dict[str, dict[str, str]]:
    return {
        row.get("sample_id", ""): row
        for row in load_csv(run_dir / "human_review_candidates_calibrated.csv")
        if row.get("sample_id")
    }


def score_map(run_dir: Path) -> dict[str, dict[str, Any]]:
    records = load_jsonl(run_dir / "ragas_eval_joined.jsonl")
    if not records:
        records = load_jsonl(run_dir / "ragas_scores.jsonl")
    return {str(row.get("sample_id")): row for row in records}


def noise_map(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in load_csv(path):
        sid = row.get("sample_id", "")
        if sid:
            out[sid] = row
    return out


def _noise_status(sid: str, noise: dict[str, dict[str, str]]) -> str:
    row = noise.get(sid)
    if not row:
        return "not_in_noise_ledger"
    if str(row.get("should_count_as_p0", "")).strip().lower() in {"false", "0", "no"}:
        return f"excluded:{row.get('issue_type', '')}"
    return f"counts:{row.get('issue_type', '')}"


def _candidate_flags(row: dict[str, str] | None, sid: str,
                     noise: dict[str, dict[str, str]]) -> dict[str, Any]:
    if not row:
        return {
            "raw_p0": False,
            "calibrated_p0": False,
            "noise_adjusted": False,
            "failure_type": "",
            "calibrated_reason": "",
        }
    raw = row.get("review_priority") == "P0"
    cal = row.get("calibrated_priority") == "P0"
    excluded = (
        sid in noise
        and str(noise[sid].get("should_count_as_p0", "")).strip().lower() in {"false", "0", "no"}
    )
    return {
        "raw_p0": raw,
        "calibrated_p0": cal,
        "noise_adjusted": cal and not excluded,
        "failure_type": row.get("calibrated_issue_type") or row.get("suspected_issue_type", ""),
        "calibrated_reason": row.get("calibrated_issue_type", ""),
    }


def _score_fields(record: dict[str, Any] | None, expected_route: str) -> dict[str, Any]:
    if not record:
        return {
            "citations_count": None,
            "route_match": None,
            "doc_hit": None,
            "section_hit": None,
            "faithfulness": None,
            "answer_relevancy": None,
            "actual_route": "",
        }
    scores = record.get("ragas_scores") or {}
    actual_route = str(record.get("route") or "")
    return {
        "citations_count": _as_int(record.get("citation_count")),
        "route_match": actual_route == expected_route if expected_route else None,
        "doc_hit": _as_bool(record.get("doc_id_hit")),
        "section_hit": _as_bool(record.get("section_norm_hit")),
        "faithfulness": _as_float(scores.get("faithfulness")),
        "answer_relevancy": _as_float(scores.get("answer_relevancy")),
        "actual_route": actual_route,
    }


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phase9_dir = Path(args.phase9_dir)
    current_dir = Path(args.current_dir)
    dataset = load_dataset(Path(args.dataset))
    noise = noise_map(Path(args.noise_ledger))
    c9 = candidate_map(phase9_dir)
    ccur = candidate_map(current_dir)
    s9 = score_map(phase9_dir)
    scur = score_map(current_dir)

    rows: list[dict[str, Any]] = []
    for sid in sorted(dataset):
        item = dataset[sid]
        expected_route = str(item.get("expected_route") or "")
        flags9 = _candidate_flags(c9.get(sid), sid, noise)
        flags_cur = _candidate_flags(ccur.get(sid), sid, noise)
        score9 = _score_fields(s9.get(sid), expected_route)
        score_cur = _score_fields(scur.get(sid), expected_route)
        row = {
            "sample_id": sid,
            "question": item.get("question", ""),
            "expected_route": expected_route,
            "raw_p0_phase9": flags9["raw_p0"],
            "raw_p0_11c": flags_cur["raw_p0"],
            "calibrated_p0_phase9": flags9["calibrated_p0"],
            "calibrated_p0_11c": flags_cur["calibrated_p0"],
            "noise_adjusted_phase9": flags9["noise_adjusted"],
            "noise_adjusted_11c": flags_cur["noise_adjusted"],
            "failure_type_phase9": flags9["failure_type"],
            "failure_type_11c": flags_cur["failure_type"],
            "noise_ledger_status": _noise_status(sid, noise),
            "calibrated_reason_phase9": flags9["calibrated_reason"],
            "calibrated_reason_11c": flags_cur["calibrated_reason"],
            "citations_count_phase9": score9["citations_count"],
            "citations_count_11c": score_cur["citations_count"],
            "route_match_phase9": score9["route_match"],
            "route_match_11c": score_cur["route_match"],
            "actual_route_phase9": score9["actual_route"],
            "actual_route_11c": score_cur["actual_route"],
            "doc_hit_phase9": score9["doc_hit"],
            "doc_hit_11c": score_cur["doc_hit"],
            "section_hit_phase9": score9["section_hit"],
            "section_hit_11c": score_cur["section_hit"],
            "faithfulness_phase9": score9["faithfulness"],
            "faithfulness_11c": score_cur["faithfulness"],
            "answer_relevancy_phase9": score9["answer_relevancy"],
            "answer_relevancy_11c": score_cur["answer_relevancy"],
            "notes": "",
        }
        notes: list[str] = []
        if row["raw_p0_phase9"] != row["raw_p0_11c"]:
            notes.append("raw_p0_changed")
        if row["calibrated_p0_phase9"] != row["calibrated_p0_11c"]:
            notes.append("calibrated_p0_changed")
        if row["noise_adjusted_phase9"] != row["noise_adjusted_11c"]:
            notes.append("noise_adjusted_changed")
        if score_cur["citations_count"] == 0 and score_cur["actual_route"] == "":
            notes.append("current_api_error_or_empty")
        row["notes"] = ";".join(notes)
        rows.append(row)

    def ids(key: str) -> list[str]:
        return [r["sample_id"] for r in rows if r[key] is True]

    summary = {
        "phase9_raw_p0_ids": ids("raw_p0_phase9"),
        "current_raw_p0_ids": ids("raw_p0_11c"),
        "raw_p0_same_ids": ids("raw_p0_phase9") == ids("raw_p0_11c"),
        "phase9_calibrated_p0_ids": ids("calibrated_p0_phase9"),
        "current_calibrated_p0_ids": ids("calibrated_p0_11c"),
        "phase9_noise_adjusted_ids": ids("noise_adjusted_phase9"),
        "current_noise_adjusted_ids": ids("noise_adjusted_11c"),
    }
    summary["new_current_noise_adjusted_ids"] = sorted(
        set(summary["current_noise_adjusted_ids"]) - set(summary["phase9_noise_adjusted_ids"])
    )
    summary["missing_current_noise_adjusted_ids"] = sorted(
        set(summary["phase9_noise_adjusted_ids"]) - set(summary["current_noise_adjusted_ids"])
    )
    summary["current_calibrated_promoted_from_non_raw_ids"] = [
        r["sample_id"] for r in rows if r["calibrated_p0_11c"] and not r["raw_p0_11c"]
    ]
    summary["phase9_calibrated_promoted_from_non_raw_ids"] = [
        r["sample_id"] for r in rows if r["calibrated_p0_phase9"] and not r["raw_p0_phase9"]
    ]
    return rows, summary


FIELDS = [
    "sample_id", "question", "expected_route",
    "raw_p0_phase9", "raw_p0_11c",
    "calibrated_p0_phase9", "calibrated_p0_11c",
    "noise_adjusted_phase9", "noise_adjusted_11c",
    "failure_type_phase9", "failure_type_11c",
    "noise_ledger_status",
    "calibrated_reason_phase9", "calibrated_reason_11c",
    "citations_count_phase9", "citations_count_11c",
    "route_match_phase9", "route_match_11c",
    "actual_route_phase9", "actual_route_11c",
    "doc_hit_phase9", "doc_hit_11c",
    "section_hit_phase9", "section_hit_11c",
    "faithfulness_phase9", "faithfulness_11c",
    "answer_relevancy_phase9", "answer_relevancy_11c",
    "notes",
]


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    changed = [
        r for r in rows
        if r["raw_p0_phase9"] or r["raw_p0_11c"]
        or r["calibrated_p0_phase9"] or r["calibrated_p0_11c"]
        or r["noise_adjusted_phase9"] or r["noise_adjusted_11c"]
    ]
    lines = [
        "# Phase 11D P0 Reconciliation Diff",
        "",
        "## Summary",
        "",
        f"- Phase 9 raw P0: `{summary['phase9_raw_p0_ids']}`",
        f"- 11C raw P0: `{summary['current_raw_p0_ids']}`",
        f"- Raw P0 same IDs: `{summary['raw_p0_same_ids']}`",
        f"- Phase 9 calibrated P0: `{summary['phase9_calibrated_p0_ids']}`",
        f"- 11C rule-review P0 candidates: `{summary['current_calibrated_p0_ids']}`",
        f"- 11C promoted from non-raw P0: `{summary['current_calibrated_promoted_from_non_raw_ids']}`",
        f"- 11C new noise-adjusted IDs: `{summary['new_current_noise_adjusted_ids']}`",
        "",
        "## P0-Related Rows",
        "",
        "| sample_id | raw 9 | raw 11C | cal 9 | cal 11C | noise 9 | noise 11C | issue 9 | issue 11C | faith 9 | faith 11C | notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|",
    ]
    for r in changed:
        lines.append(
            f"| `{r['sample_id']}` | `{r['raw_p0_phase9']}` | `{r['raw_p0_11c']}` "
            f"| `{r['calibrated_p0_phase9']}` | `{r['calibrated_p0_11c']}` "
            f"| `{r['noise_adjusted_phase9']}` | `{r['noise_adjusted_11c']}` "
            f"| `{r['failure_type_phase9']}` | `{r['failure_type_11c']}` "
            f"| `{r['faithfulness_phase9']}` | `{r['faithfulness_11c']}` | {r['notes']} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.current_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = build_rows(args)

    csv_path = output_dir / "phase11d_p0_reconciliation_diff.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "phase11d_p0_reconciliation_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = output_dir / "phase11d_p0_reconciliation_diff.md"
    write_markdown(md_path, rows, summary)

    print(f"[p0_diff] wrote {csv_path}")
    print(f"[p0_diff] wrote {json_path}")
    print(f"[p0_diff] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
