from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.phase7v_fast_ab_smoke import (
    FIXTURE_PATH,
    REPORTS_DIR,
    RESULTS_DIR,
    UNITS_PATH,
    build_ab_fixture,
    load_jsonl,
    run_ab_smoke,
    validate_ab_fixture_payload,
    write_json,
)
from scripts.evaluation.phase7v_fast_citation_guard_regression import run_citation_guard_regression
from scripts.evaluation.phase7v_fast_replay_phase7u_misses import replay_phase7u_misses
from scripts.evaluation.phase7v_fast_rollback_regression import run_rollback_regression


VALIDATION_JSON = RESULTS_DIR / "validation_summary.json"
SUMMARY_REPORT = REPORTS_DIR / "phase7v_fast_summary.md"
NEW_SCRIPT_PATHS = [
    ROOT / "scripts/evaluation/phase7v_fast_build_ab_fixture.py",
    ROOT / "scripts/evaluation/phase7v_fast_replay_phase7u_misses.py",
    ROOT / "scripts/evaluation/phase7v_fast_ab_smoke.py",
    ROOT / "scripts/evaluation/phase7v_fast_citation_guard_regression.py",
    ROOT / "scripts/evaluation/phase7v_fast_rollback_regression.py",
    ROOT / "scripts/evaluation/phase7v_fast_validate.py",
]


def validate_phase7v_fast(
    *,
    fixture_path: Path = FIXTURE_PATH,
    results_dir: Path = RESULTS_DIR,
    reports_dir: Path = REPORTS_DIR,
    tests_result: str = "not_run_by_validate_script",
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    fixture_summary = build_ab_fixture(
        output_path=fixture_path,
        summary_path=results_dir / "ab_query_fixture_summary.json",
    )
    if fixture_summary["pass"] is not True:
        errors.extend(fixture_summary["errors"])
    fixture_rows = load_jsonl(fixture_path)
    fixture_parse_summary = validate_ab_fixture_payload(
        rows=fixture_rows,
        units=load_jsonl(UNITS_PATH),
        fixture_path=fixture_path,
    )
    if fixture_parse_summary["pass"] is not True:
        errors.extend(fixture_parse_summary["errors"])

    miss_summary = replay_phase7u_misses(
        output_csv=results_dir / "phase7u_merge_miss_review.csv",
        output_report=reports_dir / "phase7u_merge_miss_review.md",
    )
    ab_summary = run_ab_smoke(
        fixture_path=fixture_path,
        output_dir=results_dir,
        reports_dir=reports_dir,
    )
    citation_summary = run_citation_guard_regression(
        fixture_path=fixture_path,
        output_csv=results_dir / "citation_guard_regression.csv",
        output_report=reports_dir / "citation_guard_regression_report.md",
    )
    rollback_summary = run_rollback_regression(
        fixture_path=fixture_path,
        output_csv=results_dir / "rollback_regression.csv",
        output_report=reports_dir / "rollback_regression_report.md",
    )

    baseline = ab_summary["baseline_current"]
    patched = ab_summary["type_aware_merge_v1"]
    if fixture_summary["preview_unit_count"] != 274:
        errors.append(f"expected 274 preview units, got {fixture_summary['preview_unit_count']}")
    if "baseline_current" not in ab_summary or "type_aware_merge_v1" not in ab_summary:
        errors.append("A/B smoke did not execute both strategies")
    if patched["merge_expected_hit_at_5"] < baseline["merge_expected_hit_at_5"]:
        errors.append("patched merge_expected_hit_at_5 is lower than baseline")
    if patched["merge_expected_hit_at_5"] == baseline["merge_expected_hit_at_5"]:
        warnings.append("patched merge_expected_hit_at_5 did not improve baseline")
    if patched["merge_expected_hit_at_5"] < 0.85:
        warnings.append("patched merge_expected_hit_at_5 is below 85 percent target")
    if patched["non_table_preview_leak_count"] != 0:
        errors.append("patched non-table preview leakage is non-zero")
    if patched["formal_citation_count"] != 0:
        errors.append("patched formal citation leakage is non-zero")
    if patched["metadata_preservation_rate"] != 1.0:
        errors.append("patched metadata preservation is below 100 percent")
    if citation_summary["pass"] is not True or citation_summary["formal_citation_count"] != 0:
        errors.append(f"citation guard regression failed: {citation_summary.get('errors', [])}")
    if rollback_summary["pass"] is not True:
        errors.append(f"rollback regression failed: {rollback_summary.get('errors', [])}")
    guardrails = {
        "milvus_accessed": False,
        "official_bm25_accessed": False,
        "embedding_run": False,
        "llm_or_ragas_called": False,
        "production_table_index_built": False,
        "preview_units_upgraded": False,
        "formal_table_citation_generated": False,
        "canonical_source_resolution": False,
        "route_c_implemented": False,
    }
    validation_status = "pass" if not errors and not warnings else "pass_with_warnings"
    if errors:
        validation_status = "fail"

    validation = {
        "validation_status": validation_status,
        "pass": validation_status in {"pass", "pass_with_warnings"},
        "errors": errors,
        "warnings": warnings,
        "fixture_summary": fixture_summary,
        "phase7u_miss_summary": miss_summary,
        "ab_summary": ab_summary,
        "citation_guard_regression": citation_summary,
        "rollback_regression": rollback_summary,
        "guardrails": guardrails,
        "tests_result": tests_result,
        "recommend_phase7w": validation_status in {"pass", "pass_with_warnings"},
        "recommend_production": False,
        "recommend_direct_production_index": False,
        "recommend_extractor_rework": False,
        "route_c_status": "backlog",
    }
    write_json(results_dir / "validation_summary.json", validation)
    write_summary_report(validation, reports_dir / "phase7v_fast_summary.md")
    return validation


def write_summary_report(validation: dict[str, Any], path: Path) -> None:
    ab = validation["ab_summary"]
    baseline = ab["baseline_current"]
    patched = ab["type_aware_merge_v1"]
    lines = [
        "# Phase7V-fast Summary",
        "",
        "## 1. Modified Files",
        "",
        "- `src/synbio_rag/application/table_preview.py`",
        "- `scripts/evaluation/phase7v_fast_build_ab_fixture.py`",
        "- `scripts/evaluation/phase7v_fast_replay_phase7u_misses.py`",
        "- `scripts/evaluation/phase7v_fast_ab_smoke.py`",
        "- `scripts/evaluation/phase7v_fast_citation_guard_regression.py`",
        "- `scripts/evaluation/phase7v_fast_rollback_regression.py`",
        "- `scripts/evaluation/phase7v_fast_validate.py`",
        "- `tests/test_phase7v_fast_type_aware_merge.py`",
        "",
        "## 2. Configs",
        "",
        "- Modified configs: no",
        "",
        "## 3. Retrieval / Model Access",
        "",
        "- Milvus accessed: no",
        "- Official BM25 accessed: no",
        "- Embedding run: no",
        "- LLM / RAGAS / OCR / VLM run: no",
        "",
        "## 4. Phase7U Miss Root Cause",
        "",
        "- Phase7U misses were ordering misses: expected table units were visible in shadow top20, but same-table row units filled merge top5.",
        f"- Phase7U miss count: {validation['phase7u_miss_summary']['phase7u_miss_count']}",
        f"- Patched recovered: {validation['phase7u_miss_summary']['patched_recovered_count']}",
        "",
        "## 5. Type-Aware Merge Patch",
        "",
        "- Default behavior remains `baseline_current` unless `table_preview_type_aware_merge_enabled` or `table_preview_merge_strategy=type_aware_merge_v1` is set on the preview config.",
        "- `table_lookup` prefers `table_unit`; `row_lookup` prefers `row_unit`; `metric_lookup` prefers `cell_group_unit`; source/reference and note routes remain preview-only observations.",
        "- Existing preview metadata and formal citation blockers are preserved.",
        "",
        "## 6. A/B Metrics",
        "",
        "| strategy | hit@5 | core hit@5 | merge rate | non-table block | leaks | citations | metadata |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in (baseline, patched):
        lines.append(
            "| {strategy} | {hit_count}/{table_count} ({hit:.2%}) | {core_hit_count}/{core_count} ({core_hit:.2%}) | "
            "{merge:.2%} | {block:.2%} | {leaks} | {citations} | {metadata:.2%} |".format(
                strategy=row["strategy"],
                hit_count=row["merge_expected_hit_at_5_count"],
                table_count=row["table_query_count"],
                hit=row["merge_expected_hit_at_5"],
                core_hit_count=row["core_merge_expected_hit_at_5_count"],
                core_count=row["core_table_query_count"],
                core_hit=row["core_merge_expected_hit_at_5"],
                merge=row["table_query_merge_rate"],
                block=row["non_table_block_rate"],
                leaks=row["non_table_preview_leak_count"],
                citations=row["formal_citation_count"],
                metadata=row["metadata_preservation_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## 7. Guards",
            "",
            f"- Non-table guard: leak_count={patched['non_table_preview_leak_count']}, block_rate={patched['non_table_block_rate']:.2%}",
            f"- Citation guard: formal_citation_count={validation['citation_guard_regression']['formal_citation_count']}",
            f"- Rollback: pass={validation['rollback_regression']['pass']}, provider_called={validation['rollback_regression']['provider_called']}",
            "",
            "## 8. Tests",
            "",
            f"- {validation['tests_result']}",
            "",
            "## 9. Decision",
            "",
            f"- validation_status: {validation['validation_status']}",
            f"- Recommend Phase7W: {validation['recommend_phase7w']}",
            "- Recommend production: false",
            "- Recommend direct production index: false",
            "- Recommend extractor rework: false",
            "- Route C: backlog",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase7V-fast outputs.")
    parser.add_argument("--fixture-path", type=_path_arg, default=FIXTURE_PATH)
    parser.add_argument("--results-dir", type=_path_arg, default=RESULTS_DIR)
    parser.add_argument("--reports-dir", type=_path_arg, default=REPORTS_DIR)
    parser.add_argument("--tests-result", default="not_run_by_validate_script")
    args = parser.parse_args()
    validation = validate_phase7v_fast(
        fixture_path=args.fixture_path,
        results_dir=args.results_dir,
        reports_dir=args.reports_dir,
        tests_result=args.tests_result,
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0 if validation["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
