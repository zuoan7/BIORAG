from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.phase7x_final_build_query_set import UNITS_PATH
from scripts.evaluation.phase7x_final_mainchain_ab_acceptance import AB_SUMMARY_PATH
from src.synbio_rag.application.generation_v2.citation_binder import CitationBinder
from src.synbio_rag.application.generation_v2.models import EvidenceCandidate, SupportItem
from src.synbio_rag.application.table_preview import adapt_table_preview_unit, apply_table_preview
from src.synbio_rag.domain.config import RetrievalConfig, Settings
from src.synbio_rag.domain.schemas import RetrievedChunk


PHASE_DIR = "v7_phase7_table_preview_final_acceptance"
RESULTS_DIR = ROOT / f"results/{PHASE_DIR}"
REPORTS_DIR = ROOT / f"reports/{PHASE_DIR}"
GUARDRAIL_CSV_PATH = RESULTS_DIR / "final_guardrail_check.csv"
GUARDRAIL_SUMMARY_PATH = RESULTS_DIR / "final_guardrail_summary.json"
GUARDRAIL_REPORT_PATH = REPORTS_DIR / "final_guardrail_report.md"

FIELDNAMES = ["check_name", "status", "details"]


class ForbiddenProvider:
    def __init__(self) -> None:
        self.called = False

    def search(self, *args: Any, **kwargs: Any) -> list[Any]:
        self.called = True
        raise AssertionError("provider must not run when TABLE_PREVIEW_ENABLED=false")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_guardrail_check(
    *,
    ab_summary_path: Path = AB_SUMMARY_PATH,
    results_dir: Path = RESULTS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    ab_summary = _load_json_if_exists(ab_summary_path)

    checks.append(_check_default_on())
    checks.append(_check_table_preview_enabled_false())
    checks.append(_check_merge_enabled_false_shadow())
    checks.append(_check_citation_guard())
    checks.append(_check_preview_units_not_production_ready())
    checks.extend(_check_ab_guardrails(ab_summary))
    checks.extend(_check_forbidden_paths_clean())
    checks.append(_check_production_table_index_not_built())

    failed = [row for row in checks if row["status"] != "pass"]
    summary = {
        "validation_status": "pass" if not failed else "fail",
        "pass": not failed,
        "failed_checks": [row["check_name"] for row in failed],
        "check_count": len(checks),
        "passed_count": len(checks) - len(failed),
        "records_path": str(results_dir / "final_guardrail_check.csv"),
        "report_path": str(reports_dir / "final_guardrail_report.md"),
    }
    write_csv(results_dir / "final_guardrail_check.csv", checks)
    write_json(results_dir / "final_guardrail_summary.json", summary)
    write_guardrail_report(summary, checks, reports_dir / "final_guardrail_report.md")
    return summary


def _check_default_on() -> dict[str, Any]:
    with _temporary_table_env({}):
        config = RetrievalConfig()
        settings = Settings.from_env()
    ok = (
        config.table_preview_enabled is True
        and config.table_preview_merge_enabled is True
        and config.table_preview_merge_strategy == "type_aware_merge_v1"
        and settings.retrieval.table_preview_enabled is True
        and settings.retrieval.table_preview_merge_enabled is True
        and settings.retrieval.table_preview_merge_strategy == "type_aware_merge_v1"
    )
    return _check(
        "default_on_config",
        ok,
        (
            f"config enabled={config.table_preview_enabled}, merge={config.table_preview_merge_enabled}, "
            f"strategy={config.table_preview_merge_strategy}"
        ),
    )


def _check_table_preview_enabled_false() -> dict[str, Any]:
    with _temporary_table_env({"TABLE_PREVIEW_ENABLED": "false"}):
        settings = Settings.from_env()
        provider = ForbiddenProvider()
        output, debug = apply_table_preview(
            question="Which table reports Table 1?",
            retrieved=[_normal_chunk()],
            config=settings.retrieval,
            provider=provider,  # type: ignore[arg-type]
        )
    ok = (
        settings.retrieval.table_preview_enabled is False
        and provider.called is False
        and debug.get("enabled") is False
        and debug.get("table_branch_executed") is False
        and len(output) == 1
        and output[0].metadata.get("object_type") == "normal_chunk"
    )
    return _check(
        "table_preview_enabled_false_normal_only_restored",
        ok,
        f"provider_called={provider.called}; debug_reason={debug.get('reason')}",
    )


def _check_merge_enabled_false_shadow() -> dict[str, Any]:
    with _temporary_table_env({"TABLE_PREVIEW_MERGE_ENABLED": "false"}):
        settings = Settings.from_env()
        output, debug = apply_table_preview(
            question="Which table reports Table 1 growth parameters?",
            retrieved=[_normal_chunk()],
            config=settings.retrieval,
        )
    preview_count = sum(
        1 for chunk in output if chunk.metadata.get("object_type") == "table_index_unit"
    )
    ok = (
        settings.retrieval.table_preview_enabled is True
        and settings.retrieval.table_preview_merge_enabled is False
        and debug.get("mode") == "shadow"
        and debug.get("table_candidates_in_rerank_input") is False
        and preview_count == 0
    )
    return _check(
        "table_preview_merge_enabled_false_shadow_only",
        ok,
        f"mode={debug.get('mode')}; candidate_count={debug.get('candidate_count')}",
    )


def _check_citation_guard() -> dict[str, Any]:
    unit = _first_preview_unit()
    chunk = adapt_table_preview_unit(unit, score=0.9)
    candidate = EvidenceCandidate(
        evidence_id="E1",
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        source_file=chunk.source_file,
        title=chunk.title,
        section=chunk.section,
        text=chunk.text,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        vector_score=chunk.vector_score,
        bm25_score=chunk.bm25_score,
        rerank_score=chunk.rerank_score,
        fusion_score=chunk.fusion_score,
        metadata=dict(chunk.metadata),
        features={},
        reasons=["phase7x_guardrail"],
    )
    support = [SupportItem("E1", candidate, 0.9, ["selected_preview_table"])]
    binder = CitationBinder()
    candidates = binder.build_citation_candidates(support)
    _answer, citations, debug = binder.bind("Preview evidence [E1].", support, citation_candidates=candidates)
    debug_paths = {
        chunk.metadata.get("source_csv_path"),
        chunk.metadata.get("source_pdf_crop_path"),
        chunk.metadata.get("source_markdown_path"),
    }
    ok = (
        len(citations) == 0
        and candidates[0].citation_eligible is False
        and debug["drop_reasons_by_evidence_id"]["E1"] == "table_preview_formal_citation_blocked"
        and not any(citation.source_file in debug_paths for citation in citations)
    )
    return _check(
        "formal_table_citation_guard",
        ok,
        f"citation_count={len(citations)}; drop={debug['drop_reasons_by_evidence_id'].get('E1')}",
    )


def _check_preview_units_not_production_ready() -> dict[str, Any]:
    units = _load_jsonl(UNITS_PATH)
    production_ready_count = sum(
        1 for unit in units if (unit.get("guardrail") or {}).get("production_ready") is True
    )
    non_preview_status_count = sum(
        1
        for unit in units
        if (unit.get("guardrail") or {}).get("index_unit_status") != "preview_only"
    )
    ok = production_ready_count == 0 and non_preview_status_count == 0
    return _check(
        "preview_evidence_not_production_ready",
        ok,
        (
            f"unit_count={len(units)}; production_ready_count={production_ready_count}; "
            f"non_preview_status_count={non_preview_status_count}"
        ),
    )


def _check_ab_guardrails(ab_summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _check(
            "non_table_preview_leak_zero",
            int(ab_summary.get("non_table_preview_leak_count", -1)) == 0,
            f"count={ab_summary.get('non_table_preview_leak_count')}",
        ),
        _check(
            "formal_table_citation_count_zero",
            int(ab_summary.get("formal_table_citation_count", -1)) == 0,
            f"count={ab_summary.get('formal_table_citation_count')}",
        ),
        _check(
            "csv_crop_path_not_formal_citation_source",
            int(ab_summary.get("csv_crop_formal_citation_leak_count", -1)) == 0,
            f"count={ab_summary.get('csv_crop_formal_citation_leak_count')}",
        ),
        _check(
            "flag_off_normal_only_restored",
            bool(ab_summary.get("flag_off_restored")) is True,
            f"flag_off_restored={ab_summary.get('flag_off_restored')}",
        ),
        _check(
            "metadata_preservation_100",
            float(ab_summary.get("metadata_preservation_rate", 0.0)) == 1.0,
            f"rate={ab_summary.get('metadata_preservation_rate')}",
        ),
    ]


def _check_forbidden_paths_clean() -> list[dict[str, Any]]:
    path_groups = {
        "milvus_not_written": ["runtime/vectorstores/milvus"],
        "bm25_not_rebuilt": ["data/paper_round1/chunks/bm25_index.json"],
        "official_chunks_not_modified": ["data/paper_round1/chunks"],
        "ingestion_pipeline_not_modified": [
            "src/synbio_rag/infrastructure",
            "src/synbio_rag/ingestion",
            "scripts/ingestion",
        ],
        "official_baseline_not_modified": ["configs", "config"],
    }
    checks = []
    for check_name, paths in path_groups.items():
        status_output = _git_status(paths)
        checks.append(_check(check_name, status_output.strip() == "", status_output.strip() or "clean"))
    return checks


def _check_production_table_index_not_built() -> dict[str, Any]:
    production_paths = [
        ROOT / "data/experiments/v7_phase7_table_index_production",
        ROOT / "results/v7_phase7_table_index_production",
        ROOT / "runtime/table_index_production",
    ]
    existing = [str(path.relative_to(ROOT)) for path in production_paths if path.exists()]
    return _check(
        "production_table_index_not_built",
        not existing,
        "existing=" + ",".join(existing) if existing else "no production table index path",
    )


def write_guardrail_report(
    summary: dict[str, Any],
    checks: list[dict[str, Any]],
    path: Path,
) -> None:
    lines = [
        "# Phase7X Final Guardrail Report",
        "",
        f"- validation_status: {summary['validation_status']}",
        f"- passed_count: {summary['passed_count']}/{summary['check_count']}",
        "",
        "## Checks",
        "",
    ]
    for row in checks:
        lines.append(f"- {row['check_name']}: {row['status']} ({row['details']})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _check(check_name: str, ok: bool, details: str) -> dict[str, Any]:
    return {"check_name": check_name, "status": "pass" if ok else "fail", "details": details}


def _git_status(paths: list[str]) -> str:
    result = subprocess.run(
        ["git", "status", "--short", "--", *paths],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return result.stderr.strip() or f"git status failed with {result.returncode}"
    return result.stdout


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _first_preview_unit() -> dict[str, Any]:
    return _load_jsonl(UNITS_PATH)[0]


def _normal_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="normal::1",
        doc_id="normal_doc",
        source_file="normal.pdf",
        title="Normal",
        section="Abstract",
        text="Normal evidence.",
        metadata={"object_type": "normal_chunk"},
    )


@contextmanager
def _temporary_table_env(values: dict[str, str]) -> Iterator[None]:
    old_values = {
        key: os.environ[key]
        for key in list(os.environ)
        if key.startswith("TABLE_PREVIEW_")
    }
    for key in list(os.environ):
        if key.startswith("TABLE_PREVIEW_"):
            del os.environ[key]
    os.environ.update(values)
    try:
        yield
    finally:
        for key in list(os.environ):
            if key.startswith("TABLE_PREVIEW_"):
                del os.environ[key]
        os.environ.update(old_values)


def _path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase7X final guardrail checks.")
    parser.add_argument("--ab-summary-path", type=_path_arg, default=AB_SUMMARY_PATH)
    parser.add_argument("--results-dir", type=_path_arg, default=RESULTS_DIR)
    parser.add_argument("--reports-dir", type=_path_arg, default=REPORTS_DIR)
    args = parser.parse_args()
    summary = run_guardrail_check(
        ab_summary_path=args.ab_summary_path,
        results_dir=args.results_dir,
        reports_dir=args.reports_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
