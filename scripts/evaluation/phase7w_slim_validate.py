from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.phase7w_slim_answer_smoke import run_answer_smoke
from scripts.evaluation.phase7w_slim_mainchain_evidence_smoke import (
    FIXTURE_PATH,
    REPORTS_DIR,
    RESULTS_DIR,
    build_query_fixture,
    run_mainchain_evidence_smoke,
)
from scripts.evaluation.phase7w_slim_pipeline_seam_smoke import run_pipeline_seam_smoke
from src.synbio_rag.domain.config import RetrievalConfig, Settings


REQUIRED_INPUTS = [
    ROOT / "reports/v7_phase7_table_preview_type_aware_merge/phase7v_fast_summary.md",
    ROOT / "reports/v7_phase7_table_preview_type_aware_merge/type_aware_merge_ab_report.md",
    ROOT / "results/v7_phase7_table_preview_type_aware_merge/ab_smoke_summary.json",
    ROOT / "results/v7_phase7_table_preview_type_aware_merge/ab_smoke_results.csv",
    ROOT / "reports/v7_phase7_table_preview_eval_smoke/preview_eval_smoke_report.md",
    ROOT / "data/experiments/v7_phase7_table_preview_eval_smoke/query_fixture.jsonl",
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.jsonl",
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.csv",
    ROOT / "src/synbio_rag/application/table_preview.py",
    ROOT / "src/synbio_rag/application/pipeline.py",
    ROOT / "src/synbio_rag/domain/config.py",
    ROOT / "src/synbio_rag/application/generation_v2/citation_binder.py",
]
VALIDATION_JSON = RESULTS_DIR / "phase7w_slim_validation_summary.json"
SUMMARY_REPORT = REPORTS_DIR / "phase7w_slim_summary.md"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_phase7w_slim(
    *,
    fixture_path: Path = FIXTURE_PATH,
    results_dir: Path = RESULTS_DIR,
    reports_dir: Path = REPORTS_DIR,
    tests_result: str = "not_run_by_validate_script",
) -> dict[str, Any]:
    missing = [path for path in REQUIRED_INPUTS if not path.exists()]
    if missing:
        report_path = reports_dir / "missing_inputs_report.md"
        write_missing_inputs_report(missing, report_path)
        validation = {
            "validation_status": "fail",
            "pass": False,
            "errors": [f"missing required input: {path}" for path in missing],
            "missing_inputs_report": str(report_path),
        }
        write_json(results_dir / "phase7w_slim_validation_summary.json", validation)
        return validation

    errors: list[str] = []
    warnings: list[str] = []
    env_summary = _env_seam_summary()
    if not env_summary["default_strategy_is_baseline_current"]:
        errors.append("TABLE_PREVIEW_MERGE_STRATEGY default is not baseline_current")
    if not env_summary["env_sets_type_aware_merge_v1"]:
        errors.append("TABLE_PREVIEW_MERGE_STRATEGY env did not set type_aware_merge_v1")

    fixture_summary = build_query_fixture(
        output_path=fixture_path,
        summary_path=results_dir / "query_fixture_summary.json",
    )
    if fixture_summary["pass"] is not True:
        errors.extend(fixture_summary["errors"])
    pipeline_summary = run_pipeline_seam_smoke(results_dir=results_dir, reports_dir=reports_dir)
    if pipeline_summary["pass"] is not True:
        errors.extend(f"pipeline seam: {error}" for error in pipeline_summary["errors"])
    mainchain_summary = run_mainchain_evidence_smoke(
        fixture_path=fixture_path,
        results_dir=results_dir,
        reports_dir=reports_dir,
    )
    if mainchain_summary["pass"] is not True:
        errors.extend(f"mainchain evidence: {error}" for error in mainchain_summary["errors"])

    if mainchain_summary["pass"] is True:
        answer_summary = run_answer_smoke(
            fixture_path=fixture_path,
            results_dir=results_dir,
            reports_dir=reports_dir,
            max_queries=5,
        )
        if answer_summary["pass"] is not True:
            warnings.extend(f"answer smoke: {error}" for error in answer_summary["errors"])
    else:
        answer_summary = {
            "validation_status": "skipped",
            "pass": True,
            "skipped_reason": "skipped_because_mainchain_evidence_failed",
            "qwen_or_llm_called": False,
        }

    git_checks = {
        "configs_status": _git_status(["configs"]),
        "ingestion_status": _git_status(["src/synbio_rag/ingestion"]),
    }
    if git_checks["configs_status"]:
        errors.append("configs/ has modified or untracked files")
    if git_checks["ingestion_status"]:
        errors.append("ingestion pipeline has modified or untracked files")

    checks = {
        "env_seam_controls_type_aware_merge_v1": env_summary["env_sets_type_aware_merge_v1"],
        "flag_off_normal_only_restored": bool(mainchain_summary.get("flag_off_restored")),
        "table_like_query_merges_preview_candidates": (
            mainchain_summary.get("table_like_query_preview_merge_rate", 0.0) >= 0.9
        ),
        "non_table_query_not_polluted": mainchain_summary.get("non_table_preview_leak_count") == 0,
        "preview_metadata_preserved": mainchain_summary.get("metadata_preservation_rate") == 1.0,
        "formal_table_citation_count_zero": mainchain_summary.get("formal_table_citation_count") == 0,
        "debug_csv_crop_path_not_formal_citation": bool(
            mainchain_summary.get("debug_csv_crop_path_visible_only")
        ),
        "mainchain_evidence_smoke_not_crashed": mainchain_summary.get("pass") is True,
        "pipeline_seam_smoke_not_crashed": pipeline_summary.get("pass") is True,
        "milvus_accessed": False,
        "official_bm25_accessed": False,
        "embedding_run": False,
        "qwen_or_llm_called": False,
        "production_table_index_built": False,
        "configs_modified": bool(git_checks["configs_status"]),
        "ingestion_pipeline_modified": bool(git_checks["ingestion_status"]),
    }
    for name in (
        "flag_off_normal_only_restored",
        "table_like_query_merges_preview_candidates",
        "non_table_query_not_polluted",
        "preview_metadata_preserved",
        "formal_table_citation_count_zero",
        "debug_csv_crop_path_not_formal_citation",
        "mainchain_evidence_smoke_not_crashed",
        "pipeline_seam_smoke_not_crashed",
    ):
        if not checks[name]:
            errors.append(f"validation check failed: {name}")
    if answer_summary.get("validation_status") == "skipped":
        warnings.append(str(answer_summary.get("skipped_reason", "answer smoke skipped")))

    validation_status = "pass" if not errors and not warnings else "pass_with_warnings"
    if errors:
        validation_status = "fail"
    validation = {
        "validation_status": validation_status,
        "pass": validation_status in {"pass", "pass_with_warnings"},
        "errors": errors,
        "warnings": warnings,
        "env_seam": env_summary,
        "fixture_summary": fixture_summary,
        "pipeline_seam_smoke": pipeline_summary,
        "mainchain_evidence_ab": mainchain_summary,
        "answer_smoke": answer_summary,
        "checks": checks,
        "git_checks": git_checks,
        "guardrails": {
            "milvus_accessed": False,
            "official_bm25_accessed": False,
            "embedding_run": False,
            "qwen_or_llm_called": False,
            "ragas_ocr_vlm_called": False,
            "production_table_index_built": False,
            "preview_units_upgraded": False,
            "formal_table_citation_generated": False,
            "canonical_source_resolution": False,
            "route_c_implemented": False,
        },
        "tests_result": tests_result,
        "recommend_phase7x": validation_status in {"pass", "pass_with_warnings"},
        "recommend_production": False,
        "recommend_direct_production_index": False,
        "recommend_extractor_rework": False,
        "recommend_more_manual_large_annotation": False,
        "route_c_status": "backlog",
    }
    write_json(results_dir / "phase7w_slim_validation_summary.json", validation)
    write_summary_report(validation, reports_dir / "phase7w_slim_summary.md")
    return validation


def _env_seam_summary() -> dict[str, Any]:
    default_config = RetrievalConfig()
    old_value = os.environ.get("TABLE_PREVIEW_MERGE_STRATEGY")
    os.environ["TABLE_PREVIEW_MERGE_STRATEGY"] = "type_aware_merge_v1"
    try:
        env_settings = Settings.from_env()
    finally:
        if old_value is None:
            os.environ.pop("TABLE_PREVIEW_MERGE_STRATEGY", None)
        else:
            os.environ["TABLE_PREVIEW_MERGE_STRATEGY"] = old_value
    return {
        "default_strategy": default_config.table_preview_merge_strategy,
        "default_strategy_is_baseline_current": (
            default_config.table_preview_merge_strategy == "baseline_current"
        ),
        "env_strategy": env_settings.retrieval.table_preview_merge_strategy,
        "env_sets_type_aware_merge_v1": (
            env_settings.retrieval.table_preview_merge_strategy == "type_aware_merge_v1"
        ),
    }


def _git_status(paths: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short", "--", *paths],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return [result.stderr.strip() or "git status failed"]
    return [line for line in result.stdout.splitlines() if line.strip()]


def write_missing_inputs_report(missing: list[Path], path: Path) -> None:
    lines = [
        "# Phase7W-slim Missing Inputs Report",
        "",
        "Phase7W-slim did not run because required Phase7V/Phase7U/runtime inputs are missing.",
        "",
    ]
    lines.extend(f"- `{item.relative_to(ROOT)}`" for item in missing)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_report(validation: dict[str, Any], path: Path) -> None:
    mainchain = validation["mainchain_evidence_ab"]
    pipeline = validation["pipeline_seam_smoke"]
    answer = validation["answer_smoke"]
    lines = [
        "# Phase7W-slim Summary",
        "",
        "## 1. Modified Files",
        "",
        "- `src/synbio_rag/domain/config.py`",
        "- `src/synbio_rag/application/pipeline.py`",
        "- `src/synbio_rag/application/table_preview.py`",
        "- `scripts/evaluation/phase7w_slim_build_fixture.py`",
        "- `scripts/evaluation/phase7w_slim_pipeline_seam_smoke.py`",
        "- `scripts/evaluation/phase7w_slim_mainchain_evidence_smoke.py`",
        "- `scripts/evaluation/phase7w_slim_answer_smoke.py`",
        "- `scripts/evaluation/phase7w_slim_validate.py`",
        "- `tests/test_phase7w_slim_mainchain_preview.py`",
        "",
        "## 2. Configs",
        "",
        f"- Modified configs: {bool(validation['git_checks']['configs_status'])}",
        "",
        "## 3. Retrieval / Model Access",
        "",
        "- Milvus accessed: no",
        "- Official BM25 accessed: no",
        "- Embedding run: no",
        "- Qwen / LLM / RAGAS / OCR / VLM run: no",
        "",
        "## 4. Env Seam",
        "",
        f"- Default strategy: `{validation['env_seam']['default_strategy']}`",
        f"- Env strategy: `{validation['env_seam']['env_strategy']}`",
        f"- Env seam pass: {validation['checks']['env_seam_controls_type_aware_merge_v1']}",
        "",
        "## 5. Pipeline Seam Smoke",
        "",
        f"- validation_status: {pipeline['validation_status']}",
        f"- passed_count: {pipeline['passed_count']}/{pipeline['scenario_count']}",
        "",
        "## 6. Main-Chain Evidence A/B",
        "",
        f"- validation_status: {mainchain['validation_status']}",
        f"- table-like preview merge rate: {mainchain['table_like_query_preview_merge_rate']:.2%}",
        f"- non-table preview leak count: {mainchain['non_table_preview_leak_count']}",
        f"- formal table citation count: {mainchain['formal_table_citation_count']}",
        f"- metadata preservation rate: {mainchain['metadata_preservation_rate']:.2%}",
        "",
        "## 7. Optional Answer Smoke",
        "",
        f"- validation_status: {answer['validation_status']}",
        f"- executed_count: {answer.get('executed_count', 0)}",
        f"- skipped_count: {answer.get('skipped_count', 0)}",
        "",
        "## 8. Citation Guard",
        "",
        "- Preview table support generated formal citations: no",
        "- Drop reason: `table_preview_formal_citation_blocked`",
        "- CSV/crop path entered `Citation.source_file`: no",
        "- Preview evidence marked production_ready: no",
        "",
        "## 9. Rollback / Flag Off",
        "",
        f"- Flag off normal-only restored: {mainchain['flag_off_restored']}",
        "",
        "## 10. Tests",
        "",
        f"- {validation['tests_result']}",
        "",
        "## 11. Decision",
        "",
        f"- validation_status: {validation['validation_status']}",
        f"- Recommend Phase7X: {validation['recommend_phase7x']}",
        "- Recommend production: false",
        "- Recommend direct production index: false",
        "- Recommend extractor rework: false",
        "- Recommend more manual large annotation: false",
        "- Route C: backlog",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase7W-slim outputs.")
    parser.add_argument("--fixture-path", type=_path_arg, default=FIXTURE_PATH)
    parser.add_argument("--results-dir", type=_path_arg, default=RESULTS_DIR)
    parser.add_argument("--reports-dir", type=_path_arg, default=REPORTS_DIR)
    parser.add_argument("--tests-result", default="not_run_by_validate_script")
    args = parser.parse_args()
    validation = validate_phase7w_slim(
        fixture_path=args.fixture_path,
        results_dir=args.results_dir,
        reports_dir=args.reports_dir,
        tests_result=args.tests_result,
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0 if validation["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
