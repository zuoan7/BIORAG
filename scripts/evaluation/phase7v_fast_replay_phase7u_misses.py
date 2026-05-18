from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.phase7v_fast_ab_smoke import (
    RESULTS_DIR,
    REPORTS_DIR,
    UNITS_PATH,
    load_jsonl,
    normal_retrieved,
    preview_chunks,
    preview_config,
    write_csv,
)
from src.synbio_rag.application.table_preview import TablePreviewCandidateProvider, apply_table_preview


PHASE7U_FIXTURE = ROOT / "data/experiments/v7_phase7_table_preview_eval_smoke/query_fixture.jsonl"
PHASE7U_RESULTS = ROOT / "results/v7_phase7_table_preview_eval_smoke"
OUTPUT_CSV = RESULTS_DIR / "phase7u_merge_miss_review.csv"
OUTPUT_REPORT = REPORTS_DIR / "phase7u_merge_miss_review.md"


def replay_phase7u_misses(
    *,
    fixture_path: Path = PHASE7U_FIXTURE,
    phase7u_results_dir: Path = PHASE7U_RESULTS,
    output_csv: Path = OUTPUT_CSV,
    output_report: Path = OUTPUT_REPORT,
) -> dict[str, Any]:
    _ensure_phase7u_records(fixture_path=fixture_path, results_dir=phase7u_results_dir)
    queries = {row["query_id"]: row for row in load_jsonl(fixture_path)}
    merge_records = load_jsonl(phase7u_results_dir / "merge_smoke_records.jsonl")
    units = load_jsonl(UNITS_PATH)
    unit_by_id = {unit.get("table_index_unit_id", ""): unit for unit in units}
    provider = TablePreviewCandidateProvider(str(UNITS_PATH))
    patched_config = preview_config(enabled=True, merge_enabled=True, strategy="type_aware_merge_v1")
    rows: list[dict[str, Any]] = []

    for record in merge_records:
        if not record.get("is_table_query"):
            continue
        if record.get("expected_candidate_merged") is True:
            continue
        query = queries[record["query_id"]]
        expected_unit_id = query.get("expected_table_index_unit_id", "")
        candidates = provider.search(query["query_text"], top_k=20)
        shadow_top = [_candidate_label(candidate) for candidate in candidates[:10]]
        expected_shadow_rank = _expected_rank(expected_unit_id, candidates)
        patched_output, patched_debug = apply_table_preview(
            question=query["query_text"],
            retrieved=normal_retrieved(query["query_id"]),
            config=patched_config,
        )
        patched_ids = [
            chunk.metadata.get("table_index_unit_id", "") for chunk in preview_chunks(patched_output)
        ]
        merge_ids = list(record.get("merged_table_index_unit_ids", []))
        miss_reason, detail = _classify_miss_reason(
            query=query,
            merge_ids=merge_ids,
            candidates=candidates,
            unit_by_id=unit_by_id,
        )
        rows.append(
            {
                "query_id": query["query_id"],
                "query_type": query["query_type"],
                "query": query["query_text"],
                "expected_table_index_unit_id": expected_unit_id,
                "expected_unit_type": query.get("expected_unit_type", ""),
                "expected_shadow_rank": expected_shadow_rank or "",
                "shadow_top_candidates": ";".join(shadow_top),
                "merge_top_candidates": ";".join(
                    _unit_label(unit_id, unit_by_id) for unit_id in merge_ids
                ),
                "miss_reason": miss_reason,
                "miss_reason_detail": detail,
                "patched_merge_strategy": patched_debug.get("merge_strategy", ""),
                "patched_top_candidates": ";".join(
                    _unit_label(unit_id, unit_by_id) for unit_id in patched_ids
                ),
                "patched_expected_hit_at_5": expected_unit_id in patched_ids,
            }
        )

    write_csv(
        output_csv,
        rows,
        [
            "query_id",
            "query_type",
            "query",
            "expected_table_index_unit_id",
            "expected_unit_type",
            "expected_shadow_rank",
            "shadow_top_candidates",
            "merge_top_candidates",
            "miss_reason",
            "miss_reason_detail",
            "patched_merge_strategy",
            "patched_top_candidates",
            "patched_expected_hit_at_5",
        ],
    )
    _write_report(rows, output_report)
    summary = {
        "phase7u_miss_count": len(rows),
        "patched_recovered_count": sum(1 for row in rows if row["patched_expected_hit_at_5"]),
        "output_csv": str(output_csv),
        "output_report": str(output_report),
    }
    return summary


def _ensure_phase7u_records(*, fixture_path: Path, results_dir: Path) -> None:
    if (
        fixture_path.exists()
        and (results_dir / "merge_smoke_records.jsonl").exists()
        and (results_dir / "shadow_smoke_records.jsonl").exists()
    ):
        return
    from scripts.evaluation.phase7u_shadow_smoke import build_query_fixture, run_merge_smoke, run_shadow_smoke

    if not fixture_path.exists():
        build_query_fixture(output_path=fixture_path, summary_path=results_dir / "query_fixture_summary.json")
    run_shadow_smoke(fixture_path=fixture_path, output_dir=results_dir)
    run_merge_smoke(fixture_path=fixture_path, output_dir=results_dir)


def _candidate_label(candidate: Any) -> str:
    metadata = candidate.chunk.metadata
    return "{rank}:{unit_id}:{unit_type}:{score}".format(
        rank=candidate.rank,
        unit_id=metadata.get("table_index_unit_id", ""),
        unit_type=metadata.get("table_unit_type", ""),
        score=round(candidate.score, 6),
    )


def _unit_label(unit_id: str, unit_by_id: dict[str, dict[str, Any]]) -> str:
    unit = unit_by_id.get(unit_id, {})
    return f"{unit_id}:{unit.get('unit_type', '')}"


def _expected_rank(expected_unit_id: str, candidates: list[Any]) -> int | None:
    for idx, candidate in enumerate(candidates, start=1):
        if candidate.chunk.metadata.get("table_index_unit_id") == expected_unit_id:
            return idx
    return None


def _classify_miss_reason(
    *,
    query: dict[str, Any],
    merge_ids: list[str],
    candidates: list[Any],
    unit_by_id: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    expected_unit_type = query.get("expected_unit_type", "")
    expected_unit_id = query.get("expected_table_index_unit_id", "")
    merge_unit_types = [unit_by_id.get(unit_id, {}).get("unit_type", "") for unit_id in merge_ids]
    expected_rank = _expected_rank(expected_unit_id, candidates)
    expected_score = None
    top5_min_score = None
    scores = [candidate.score for candidate in candidates[:5]]
    if scores:
        top5_min_score = min(scores)
    for candidate in candidates:
        if candidate.chunk.metadata.get("table_index_unit_id") == expected_unit_id:
            expected_score = candidate.score
            break
    if expected_unit_type == "table_unit" and "table_unit" not in merge_unit_types:
        return (
            "table_unit_under_boosted",
            "table_lookup target was present in shadow top20 but same-table row units filled merge top5",
        )
    if expected_unit_type and expected_unit_type not in merge_unit_types:
        return (
            "wrong_unit_type_preferred",
            f"merge top5 unit types were {merge_unit_types}, expected {expected_unit_type}",
        )
    if expected_score is not None and top5_min_score is not None and abs(top5_min_score - expected_score) <= 0.05:
        return (
            "score_tie_or_near_tie",
            f"expected rank={expected_rank}, expected_score={expected_score:.6f}, top5_min={top5_min_score:.6f}",
        )
    return (
        "insufficient_diversity_cap",
        f"expected rank={expected_rank}, merge_ids={merge_ids}",
    )


def _write_report(rows: list[dict[str, Any]], path: Path) -> None:
    reason_counts = Counter(row["miss_reason"] for row in rows)
    lines = [
        "# Phase7U Merge Miss Review",
        "",
        "Phase7U misses are table-preview ordering misses, not loader misses: the expected units were visible in shadow top20 but absent from merge top5.",
        "",
        "## Root Cause Counts",
        "",
    ]
    for reason, count in sorted(reason_counts.items()):
        lines.append(f"- {reason}: {count}")
    lines.extend(["", "## Misses", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['query_id']}",
                "",
                f"- query_type: {row['query_type']}",
                f"- expected: {row['expected_table_index_unit_id']} ({row['expected_unit_type']})",
                f"- expected_shadow_rank: {row['expected_shadow_rank']}",
                f"- miss_reason: {row['miss_reason']}",
                f"- patched_expected_hit_at_5: {row['patched_expected_hit_at_5']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay Phase7U merge misses for Phase7V-fast.")
    parser.add_argument("--fixture-path", type=_path_arg, default=PHASE7U_FIXTURE)
    parser.add_argument("--phase7u-results-dir", type=_path_arg, default=PHASE7U_RESULTS)
    parser.add_argument("--output-csv", type=_path_arg, default=OUTPUT_CSV)
    parser.add_argument("--output-report", type=_path_arg, default=OUTPUT_REPORT)
    args = parser.parse_args()
    summary = replay_phase7u_misses(
        fixture_path=args.fixture_path,
        phase7u_results_dir=args.phase7u_results_dir,
        output_csv=args.output_csv,
        output_report=args.output_report,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
