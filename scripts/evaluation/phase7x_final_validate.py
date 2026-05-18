from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.phase7x_final_answer_acceptance import (
    ANSWER_SUMMARY_PATH,
    REVIEW_CARDS_PATH,
)
from scripts.evaluation.phase7x_final_build_query_set import RESULTS_DIR as QUERY_RESULTS_DIR
from scripts.evaluation.phase7x_final_guardrail_check import GUARDRAIL_SUMMARY_PATH
from scripts.evaluation.phase7x_final_mainchain_ab_acceptance import AB_SUMMARY_PATH
from src.synbio_rag.domain.config import RetrievalConfig


PHASE_DIR = "v7_phase7_table_preview_final_acceptance"
RESULTS_DIR = ROOT / f"results/{PHASE_DIR}"
REPORTS_DIR = ROOT / f"reports/{PHASE_DIR}"
QUERY_SUMMARY_PATH = QUERY_RESULTS_DIR / "final_acceptance_query_set_summary.json"
FINAL_SUMMARY_PATH = RESULTS_DIR / "final_acceptance_summary.json"
FINAL_REPORT_PATH = REPORTS_DIR / "phase7x_final_summary.md"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_phase7x_final(
    *,
    query_summary_path: Path = QUERY_SUMMARY_PATH,
    ab_summary_path: Path = AB_SUMMARY_PATH,
    answer_summary_path: Path = ANSWER_SUMMARY_PATH,
    guardrail_summary_path: Path = GUARDRAIL_SUMMARY_PATH,
    review_cards_path: Path = REVIEW_CARDS_PATH,
    tests_result: str = "not_run",
    results_dir: Path = RESULTS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    query_summary = _load_json(query_summary_path)
    ab_summary = _load_json(ab_summary_path)
    answer_summary = _load_json(answer_summary_path)
    guardrail_summary = _load_json(guardrail_summary_path)
    config = RetrievalConfig()

    answer_labels = answer_summary.get("answer_improvement_counts", {})
    pass_conditions = {
        "default_on_config": (
            config.table_preview_enabled is True
            and config.table_preview_merge_enabled is True
            and config.table_preview_merge_strategy == "type_aware_merge_v1"
        ),
        "emergency_off_guardrail_passed": bool(guardrail_summary.get("pass")),
        "query_count_distribution_valid": bool(query_summary.get("pass")),
        "mainchain_ab_real_backend": ab_summary.get("backend_mode") == "real",
        "mainchain_ab_passed": bool(ab_summary.get("pass")),
        "table_like_preview_support_rate_ge_80": float(
            ab_summary.get("table_like_preview_support_rate", 0.0)
        )
        >= 0.8,
        "evidence_better_or_same_rate_ge_90": float(
            ab_summary.get("evidence_better_or_same_rate", 0.0)
        )
        >= 0.9,
        "answer_smoke_not_crashed": bool(answer_summary.get("pass")),
        "table_like_answer_preview_better_gt_worse": int(answer_labels.get("preview_better", 0))
        > int(answer_labels.get("preview_worse", 0)),
        "non_table_preview_leak_zero": int(ab_summary.get("non_table_preview_leak_count", -1)) == 0,
        "formal_table_citation_count_zero": int(
            ab_summary.get("formal_table_citation_count", -1)
        )
        == 0
        and int(answer_summary.get("formal_table_citation_count", -1)) == 0,
        "csv_crop_formal_citation_leak_zero": int(
            ab_summary.get("csv_crop_formal_citation_leak_count", -1)
        )
        == 0
        and int(answer_summary.get("csv_crop_formal_citation_leak_count", -1)) == 0,
        "metadata_preservation_100": float(ab_summary.get("metadata_preservation_rate", 0.0))
        == 1.0,
        "review_cards_generated": review_cards_path.exists(),
        "tests_passed": _tests_passed(tests_result),
    }
    failed_conditions = [name for name, ok in pass_conditions.items() if not ok]
    validation_status = "pass" if not failed_conditions else "fail"
    if ab_summary.get("validation_status") == "blocked" or answer_summary.get(
        "validation_status"
    ) == "skipped_provider_unavailable":
        validation_status = "blocked"

    summary = {
        "validation_status": validation_status,
        "pass": validation_status == "pass",
        "failed_conditions": failed_conditions,
        "pass_conditions": pass_conditions,
        "tests_result": tests_result,
        "modified_files": [
            "src/synbio_rag/domain/config.py",
            "src/synbio_rag/application/pipeline.py",
            "scripts/evaluation/phase7v_fast_ab_smoke.py",
            "scripts/evaluation/phase7x_final_build_query_set.py",
            "scripts/evaluation/phase7x_final_mainchain_ab_acceptance.py",
            "scripts/evaluation/phase7x_final_answer_acceptance.py",
            "scripts/evaluation/phase7x_final_guardrail_check.py",
            "scripts/evaluation/phase7x_final_validate.py",
            "tests/test_phase7t_table_preview_scaffold.py",
            "tests/test_phase7w_slim_mainchain_preview.py",
            "tests/test_phase7x_final_default_on_table_preview.py",
        ],
        "default_on_effective": pass_conditions["default_on_config"],
        "emergency_off_effective": pass_conditions["emergency_off_guardrail_passed"],
        "query_summary": query_summary,
        "mainchain_ab_summary": ab_summary,
        "answer_summary": answer_summary,
        "guardrail_summary": guardrail_summary,
        "review_cards_path": str(review_cards_path),
        "stop_table_enhancement_validation": validation_status == "pass",
        "default_mainchain_table_preview": validation_status == "pass",
        "recommend_production_table_index": "independent_followup_not_blocking",
        "recommend_extractor_rework": False,
        "route_c_status": "backlog",
        "decision": _decision(validation_status),
    }
    write_json(results_dir / "final_acceptance_summary.json", summary)
    write_final_report(summary, reports_dir / "phase7x_final_summary.md")
    return summary


def write_final_report(summary: dict[str, Any], path: Path) -> None:
    query = summary["query_summary"]
    ab = summary["mainchain_ab_summary"]
    answer = summary["answer_summary"]
    guard = summary["guardrail_summary"]
    lines = [
        "# Phase7X Final Summary",
        "",
        "## 1. Modified Files",
        "",
        *[f"- `{path_value}`" for path_value in summary["modified_files"]],
        "",
        "## 2. Default-On",
        "",
        f"- default_on_effective: {summary['default_on_effective']}",
        "- default strategy: `type_aware_merge_v1`",
        "",
        "## 3. Emergency Off",
        "",
        f"- emergency_off_effective: {summary['emergency_off_effective']}",
        f"- guardrail_status: {guard.get('validation_status')}",
        "",
        "## 4. Query Set",
        "",
        f"- query_count: {query.get('query_count')}",
        f"- table_like_query_count: {query.get('table_like_query_count')}",
        f"- non_table_control_count: {query.get('non_table_control_count')}",
        f"- query_type_counts: {query.get('query_type_counts')}",
        "",
        "## 5. Main-Chain A/B Evidence",
        "",
        f"- backend_mode: {ab.get('backend_mode')}",
        f"- real_backend_status: {ab.get('real_backend_status')}",
        f"- validation_status: {ab.get('validation_status')}",
        f"- table_like_preview_support_rate: {float(ab.get('table_like_preview_support_rate', 0.0)):.2%}",
        f"- evidence_better_or_same_rate: {float(ab.get('evidence_better_or_same_rate', 0.0)):.2%}",
        f"- evidence_improvement_counts: {ab.get('evidence_improvement_counts')}",
        "",
        "## 6. Answer Smoke",
        "",
        f"- validation_status: {answer.get('validation_status')}",
        f"- query_count: {answer.get('query_count')}",
        f"- answers_using_table_evidence_count: {answer.get('answers_using_table_evidence_count')}",
        f"- answer_improvement_counts: {answer.get('answer_improvement_counts')}",
        "",
        "## 7. Review Cards",
        "",
        f"- generated: {Path(summary['review_cards_path']).exists()}",
        f"- path: `{summary['review_cards_path']}`",
        "",
        "## 8. Non-Table Guard",
        "",
        f"- non_table_preview_leak_count: {ab.get('non_table_preview_leak_count')}",
        f"- non_table_answer_preview_leak_count: {answer.get('non_table_answer_preview_leak_count')}",
        "",
        "## 9. Citation Guard",
        "",
        f"- formal_table_citation_count: {ab.get('formal_table_citation_count')}",
        f"- csv_crop_formal_citation_leak_count: {ab.get('csv_crop_formal_citation_leak_count')}",
        "",
        "## 10. Rollback / Flag-Off",
        "",
        f"- flag_off_restored: {ab.get('flag_off_restored')}",
        "",
        "## 11. Tests",
        "",
        f"- tests_result: {summary['tests_result']}",
        "",
        "## 12. Validation",
        "",
        f"- validation_status: {summary['validation_status']}",
        f"- failed_conditions: {summary['failed_conditions']}",
        "",
        "## 13. Stop Further Table Validation",
        "",
        f"- stop_table_enhancement_validation: {summary['stop_table_enhancement_validation']}",
        "",
        "## 14. Default Main-Chain Integration",
        "",
        f"- default_mainchain_table_preview: {summary['default_mainchain_table_preview']}",
        "",
        "## 15. Follow-Up Scope",
        "",
        f"- recommend_production_table_index: {summary['recommend_production_table_index']}",
        f"- recommend_extractor_rework: {summary['recommend_extractor_rework']}",
        f"- route_c_status: {summary['route_c_status']}",
        "",
        "## Final Decision",
        "",
        summary["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decision(validation_status: str) -> str:
    if validation_status == "pass":
        return (
            "表格结构化增强主链路验收通过；table preview 默认开启；不再继续表格增强验证。"
            "后续只处理 bugfix / monitoring / production index 独立议题。"
        )
    if validation_status == "blocked":
        return (
            "Phase7X-final blocked：真实主链路 backend 不可用或 answer provider 不可用，"
            "不能声明最终通过。"
        )
    return "Phase7X-final 未通过：只修具体失败点，不回到表格抽取阶段，不重开大规模验证。"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _tests_passed(tests_result: str) -> bool:
    value = tests_result.strip().lower()
    if not value or value == "not_run":
        return False
    return "failed" not in value and "error" not in value


def _path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase7X final acceptance.")
    parser.add_argument("--query-summary-path", type=_path_arg, default=QUERY_SUMMARY_PATH)
    parser.add_argument("--ab-summary-path", type=_path_arg, default=AB_SUMMARY_PATH)
    parser.add_argument("--answer-summary-path", type=_path_arg, default=ANSWER_SUMMARY_PATH)
    parser.add_argument("--guardrail-summary-path", type=_path_arg, default=GUARDRAIL_SUMMARY_PATH)
    parser.add_argument("--review-cards-path", type=_path_arg, default=REVIEW_CARDS_PATH)
    parser.add_argument("--tests-result", default="not_run")
    parser.add_argument("--results-dir", type=_path_arg, default=RESULTS_DIR)
    parser.add_argument("--reports-dir", type=_path_arg, default=REPORTS_DIR)
    args = parser.parse_args()
    summary = validate_phase7x_final(
        query_summary_path=args.query_summary_path,
        ab_summary_path=args.ab_summary_path,
        answer_summary_path=args.answer_summary_path,
        guardrail_summary_path=args.guardrail_summary_path,
        review_cards_path=args.review_cards_path,
        tests_result=args.tests_result,
        results_dir=args.results_dir,
        reports_dir=args.reports_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
