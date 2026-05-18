from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag.application.generation_v2.citation_binder import CitationBinder
from src.synbio_rag.application.generation_v2.models import EvidenceCandidate, SupportItem
from src.synbio_rag.application.table_preview import TablePreviewCandidateProvider, apply_table_preview
from src.synbio_rag.domain.config import RetrievalConfig
from src.synbio_rag.domain.schemas import RetrievedChunk


PHASE_DIR = "v7_phase7_table_preview_type_aware_merge"
UNITS_PATH = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.jsonl"
)
SOURCE_QUERY_SET = (
    ROOT / "data/experiments/v7_phase7_table_retrieval_wiring_preview/query_set.preview.jsonl"
)
FIXTURE_PATH = ROOT / f"data/experiments/{PHASE_DIR}/ab_query_fixture.jsonl"
RESULTS_DIR = ROOT / f"results/{PHASE_DIR}"
REPORTS_DIR = ROOT / f"reports/{PHASE_DIR}"

CORE_TABLE_QUERY_TYPES = {"table_lookup", "row_lookup", "metric_lookup"}
OBSERVATION_TABLE_QUERY_TYPES = {"source_or_reference_lookup", "unit_or_note_lookup"}
TABLE_QUERY_TYPES = CORE_TABLE_QUERY_TYPES | OBSERVATION_TABLE_QUERY_TYPES
STRATEGIES = ("baseline_current", "type_aware_merge_v1")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_preview_units(units_path: Path = UNITS_PATH) -> list[dict[str, Any]]:
    return load_jsonl(units_path)


def normal_retrieved(query_id: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"normal::{query_id}",
            doc_id="normal_doc",
            source_file="normal_only_stub.pdf",
            title="Normal retrieval stub",
            section="Abstract",
            text="Normal retrieval evidence stub used only for Phase7V-fast preview smoke.",
            vector_score=0.2,
            bm25_score=0.0,
            rerank_score=0.0,
            fusion_score=0.2,
            metadata={"object_type": "normal_chunk", "phase7v_stub_normal": True},
        )
    ]


def preview_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return [chunk for chunk in chunks if chunk.metadata.get("object_type") == "table_index_unit"]


def preview_config(*, enabled: bool, merge_enabled: bool, strategy: str = "baseline_current") -> RetrievalConfig:
    config = RetrievalConfig(
        table_preview_enabled=enabled,
        table_preview_units_path=str(UNITS_PATH),
        table_preview_max_candidates=20,
        table_preview_merge_enabled=merge_enabled,
        table_preview_merge_strategy=strategy,
        table_preview_merge_max_candidates=5,
        table_preview_min_score=0.05,
        table_preview_allow_formal_citation=False,
        rerank_score_floor_ratio=0.0,
    )
    if strategy == "type_aware_merge_v1":
        setattr(config, "table_preview_type_aware_merge_enabled", True)
        setattr(config, "table_preview_merge_strategy", "type_aware_merge_v1")
    return config


def build_ab_fixture(
    *,
    source_query_set: Path = SOURCE_QUERY_SET,
    units_path: Path = UNITS_PATH,
    output_path: Path = FIXTURE_PATH,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    units = load_preview_units(units_path)
    source_queries = load_jsonl(source_query_set)
    provider = TablePreviewCandidateProvider(str(units_path))
    quotas = {
        "table_lookup": 8,
        "row_lookup": 12,
        "metric_lookup": 8,
        "source_or_reference_lookup": 2,
        "unit_or_note_lookup": 2,
    }
    selected: list[dict[str, Any]] = []
    for query_type, quota in quotas.items():
        candidates = [
            row
            for row in source_queries
            if row.get("query_type") == query_type and _expected_seen_at_20(row, provider)
        ]
        selected.extend(candidates[:quota])

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        rows.append(
            {
                "query_id": f"phase7v_query_{idx:03d}",
                "source_query_id": row.get("query_id", ""),
                "query_text": row["query_text"],
                "query_type": row["query_type"],
                "expected_doc_id": row.get("expected_doc_id", ""),
                "expected_table_id": row.get("expected_table_id", ""),
                "expected_table_index_unit_id": row.get("expected_table_index_unit_id", ""),
                "expected_unit_type": row.get("expected_unit_type", ""),
                "expected_row_label": row.get("expected_row_label", ""),
                "query_notes": "phase7v fast type-aware merge A/B smoke fixture",
            }
        )

    for offset, control in enumerate(_non_table_controls(), start=len(rows) + 1):
        rows.append(
            {
                "query_id": f"phase7v_query_{offset:03d}",
                "source_query_id": "",
                "query_text": control["query_text"],
                "query_type": "non_table_control",
                "expected_doc_id": control["expected_doc_id"],
                "expected_table_id": "",
                "expected_table_index_unit_id": "",
                "expected_unit_type": "none",
                "expected_row_label": "",
                "query_notes": "non-table control with doc_id overlap for merge guard smoke",
            }
        )

    write_jsonl(output_path, rows)
    summary = validate_ab_fixture_payload(rows=rows, units=units, fixture_path=output_path)
    summary["source_query_set"] = str(source_query_set)
    summary["units_path"] = str(units_path)
    if summary_path is None:
        summary_path = RESULTS_DIR / "ab_query_fixture_summary.json"
    write_json(summary_path, summary)
    return summary


def validate_ab_fixture_payload(
    *,
    rows: list[dict[str, Any]],
    units: list[dict[str, Any]],
    fixture_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    counts = Counter(row.get("query_type", "") for row in rows)
    unit_ids = {str(unit.get("table_index_unit_id", "")) for unit in units}
    if len(units) != 274:
        errors.append(f"expected 274 preview units, got {len(units)}")
    if not 30 <= len(rows) <= 40:
        errors.append(f"expected 30-40 fixture queries, got {len(rows)}")
    if counts.get("non_table_control", 0) < 8:
        errors.append("expected at least 8 non_table_control queries")
    for query_type in CORE_TABLE_QUERY_TYPES:
        if counts.get(query_type, 0) <= 0:
            errors.append(f"missing core query type: {query_type}")
    for row in rows:
        query_type = row.get("query_type")
        if query_type in TABLE_QUERY_TYPES:
            expected_unit_id = str(row.get("expected_table_index_unit_id", ""))
            if expected_unit_id not in unit_ids:
                errors.append(f"{row.get('query_id')} expected unit not in preview units")
            if not row.get("expected_table_id"):
                errors.append(f"{row.get('query_id')} missing expected_table_id")
            if row.get("expected_unit_type") not in {
                "table_unit",
                "row_unit",
                "cell_group_unit",
            }:
                errors.append(f"{row.get('query_id')} has invalid expected_unit_type")
        elif query_type == "non_table_control":
            if row.get("expected_unit_type") != "none":
                errors.append(f"{row.get('query_id')} non-table control must use expected_unit_type=none")
        else:
            errors.append(f"{row.get('query_id')} unknown query_type={query_type!r}")
    return {
        "pass": not errors,
        "errors": errors,
        "fixture_path": str(fixture_path),
        "preview_unit_count": len(units),
        "query_count": len(rows),
        "query_type_counts": dict(sorted(counts.items())),
        "table_query_count": sum(counts.get(query_type, 0) for query_type in TABLE_QUERY_TYPES),
        "core_table_query_count": sum(counts.get(query_type, 0) for query_type in CORE_TABLE_QUERY_TYPES),
        "non_table_control_count": counts.get("non_table_control", 0),
    }


def ensure_ab_fixture(fixture_path: Path = FIXTURE_PATH) -> list[dict[str, Any]]:
    if not fixture_path.exists():
        build_ab_fixture(output_path=fixture_path)
    return load_jsonl(fixture_path)


def run_ab_smoke(
    *,
    fixture_path: Path = FIXTURE_PATH,
    output_dir: Path = RESULTS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    queries = ensure_ab_fixture(fixture_path)
    units = load_preview_units(UNITS_PATH)
    all_records: list[dict[str, Any]] = []
    strategy_summaries: dict[str, dict[str, Any]] = {}
    for strategy in STRATEGIES:
        records, summary = _run_strategy_smoke(
            strategy=strategy,
            queries=queries,
            preview_unit_load_count=len(units),
        )
        all_records.extend(records)
        strategy_summaries[strategy] = summary

    baseline = strategy_summaries["baseline_current"]
    patched = strategy_summaries["type_aware_merge_v1"]
    guard_pass = all(
        summary["non_table_preview_leak_count"] == 0
        and summary["formal_citation_count"] == 0
        and summary["metadata_preservation_rate"] == 1.0
        and summary["non_table_block_rate"] == 1.0
        for summary in strategy_summaries.values()
    )
    improved = patched["merge_expected_hit_at_5"] > baseline["merge_expected_hit_at_5"]
    non_degraded = patched["merge_expected_hit_at_5"] >= baseline["merge_expected_hit_at_5"]
    target_met = patched["merge_expected_hit_at_5"] >= 0.85
    status = "pass" if guard_pass and non_degraded and target_met else "pass_with_warnings"
    if not guard_pass or not non_degraded:
        status = "fail"

    summary = {
        "validation_status": status,
        "pass": status in {"pass", "pass_with_warnings"},
        "preview_unit_load_count": len(units),
        "query_count": len(queries),
        "table_query_count": sum(1 for row in queries if row["query_type"] in TABLE_QUERY_TYPES),
        "non_table_control_count": sum(1 for row in queries if row["query_type"] == "non_table_control"),
        "baseline_current": baseline,
        "type_aware_merge_v1": patched,
        "merge_expected_hit_at_5_lift": round(
            patched["merge_expected_hit_at_5"] - baseline["merge_expected_hit_at_5"],
            6,
        ),
        "merge_expected_hit_at_5_improved": improved,
        "merge_expected_hit_at_5_non_degraded": non_degraded,
        "target_85_percent_met": target_met,
        "guardrails": _guardrails(),
        "records_path": str(output_dir / "ab_smoke_results.csv"),
    }
    write_csv(output_dir / "ab_smoke_results.csv", all_records, _ab_record_fieldnames())
    write_json(output_dir / "ab_smoke_summary.json", summary)
    write_ab_report(summary, reports_dir / "type_aware_merge_ab_report.md")
    return summary


def _run_strategy_smoke(
    *,
    strategy: str,
    queries: list[dict[str, Any]],
    preview_unit_load_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shadow_config = preview_config(enabled=True, merge_enabled=False, strategy="baseline_current")
    merge_config = preview_config(enabled=True, merge_enabled=True, strategy=strategy)
    binder = CitationBinder()
    records: list[dict[str, Any]] = []
    metadata_ok_count = 0
    preview_chunk_count = 0
    formal_citation_count = 0

    for query in queries:
        shadow_output, shadow_debug = apply_table_preview(
            question=query["query_text"],
            retrieved=normal_retrieved(query["query_id"]),
            config=shadow_config,
        )
        if preview_chunks(shadow_output):
            raise AssertionError("shadow mode must not emit preview chunks")
        output, debug = apply_table_preview(
            question=query["query_text"],
            retrieved=normal_retrieved(query["query_id"]),
            config=merge_config,
        )
        table_output = preview_chunks(output)
        expected_unit_id = query.get("expected_table_index_unit_id", "")
        candidate_ids = shadow_debug.get("candidate_table_index_unit_ids", [])
        merged_ids = [chunk.metadata.get("table_index_unit_id", "") for chunk in table_output]
        is_table_query = query["query_type"] in TABLE_QUERY_TYPES
        chunk_metadata_ok = all(_preview_metadata_ok(chunk) for chunk in table_output)
        metadata_ok_count += sum(1 for chunk in table_output if _preview_metadata_ok(chunk))
        preview_chunk_count += len(table_output)
        citation_count = 0
        for idx, chunk in enumerate(table_output, start=1):
            citation_count += _formal_citation_count(
                binder=binder,
                chunk=chunk,
                evidence_id=f"E{idx}",
            )
        formal_citation_count += citation_count
        records.append(
            {
                "strategy": strategy,
                "query_id": query["query_id"],
                "source_query_id": query.get("source_query_id", ""),
                "query_type": query["query_type"],
                "is_table_query": is_table_query,
                "query_route": debug.get("query_route", ""),
                "mode": debug.get("mode", ""),
                "reason": debug.get("reason", ""),
                "candidate_count": debug.get("candidate_count", 0),
                "merged_count": debug.get("merged_count", 0),
                "preview_output_count": len(table_output),
                "expected_table_index_unit_id": expected_unit_id,
                "expected_unit_type": query.get("expected_unit_type", ""),
                "shadow_expected_hit_at_20": bool(expected_unit_id and expected_unit_id in candidate_ids),
                "merge_expected_hit_at_5": bool(expected_unit_id and expected_unit_id in merged_ids),
                "table_candidates_in_rerank_input": debug.get("table_candidates_in_rerank_input", False),
                "preview_metadata_preserved": chunk_metadata_ok,
                "formal_citation_count": citation_count,
                "merged_table_index_unit_ids": ";".join(str(value) for value in merged_ids),
                "merged_unit_types": ";".join(
                    str(chunk.metadata.get("table_unit_type", "")) for chunk in table_output
                ),
            }
        )

    table_records = [row for row in records if row["is_table_query"]]
    core_records = [row for row in records if row["query_type"] in CORE_TABLE_QUERY_TYPES]
    non_table_records = [row for row in records if not row["is_table_query"]]
    table_query_count = len(table_records)
    non_table_count = len(non_table_records)
    expected_hit_count = sum(1 for row in table_records if row["merge_expected_hit_at_5"])
    core_expected_hit_count = sum(1 for row in core_records if row["merge_expected_hit_at_5"])
    summary = {
        "strategy": strategy,
        "preview_unit_load_count": preview_unit_load_count,
        "query_count": len(records),
        "table_query_count": table_query_count,
        "core_table_query_count": len(core_records),
        "non_table_control_count": non_table_count,
        "shadow_expected_hit_at_20_count": sum(
            1 for row in table_records if row["shadow_expected_hit_at_20"]
        ),
        "shadow_expected_hit_at_20": _rate(
            sum(1 for row in table_records if row["shadow_expected_hit_at_20"]),
            table_query_count,
        ),
        "merge_expected_hit_at_5_count": expected_hit_count,
        "merge_expected_hit_at_5": _rate(expected_hit_count, table_query_count),
        "core_merge_expected_hit_at_5_count": core_expected_hit_count,
        "core_merge_expected_hit_at_5": _rate(core_expected_hit_count, len(core_records)),
        "table_query_merge_rate": _rate(
            sum(1 for row in table_records if int(row["preview_output_count"]) > 0),
            table_query_count,
        ),
        "non_table_block_rate": _rate(
            sum(
                1
                for row in non_table_records
                if row["mode"] == "merge_blocked" and int(row["preview_output_count"]) == 0
            ),
            non_table_count,
        ),
        "non_table_preview_leak_count": sum(
            1 for row in non_table_records if int(row["preview_output_count"]) > 0
        ),
        "formal_citation_count": formal_citation_count,
        "metadata_preservation_rate": _rate(metadata_ok_count, preview_chunk_count),
        "preview_chunk_count": preview_chunk_count,
    }
    return records, summary


def write_ab_report(summary: dict[str, Any], path: Path) -> None:
    baseline = summary["baseline_current"]
    patched = summary["type_aware_merge_v1"]
    lines = [
        "# Phase7V-fast Type-Aware Merge A/B Smoke",
        "",
        "This is a preview-only smoke. It does not build a production table index, call Milvus, query official BM25, run embeddings, or call LLM/RAGAS/OCR/VLM.",
        "",
        "## Metrics",
        "",
        "| strategy | table queries | expected hit@5 | core hit@5 | merge rate | non-table block | leaks | formal citations | metadata |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in (baseline, patched):
        lines.append(
            "| {strategy} | {table_query_count} | {hit_count}/{table_query_count} ({hit:.2%}) | "
            "{core_hit_count}/{core_count} ({core_hit:.2%}) | {merge_rate:.2%} | {block_rate:.2%} | "
            "{leaks} | {citations} | {metadata:.2%} |".format(
                strategy=row["strategy"],
                table_query_count=row["table_query_count"],
                hit_count=row["merge_expected_hit_at_5_count"],
                hit=row["merge_expected_hit_at_5"],
                core_hit_count=row["core_merge_expected_hit_at_5_count"],
                core_count=row["core_table_query_count"],
                core_hit=row["core_merge_expected_hit_at_5"],
                merge_rate=row["table_query_merge_rate"],
                block_rate=row["non_table_block_rate"],
                leaks=row["non_table_preview_leak_count"],
                citations=row["formal_citation_count"],
                metadata=row["metadata_preservation_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- validation_status: {summary['validation_status']}",
            f"- merge_expected_hit_at_5_lift: {summary['merge_expected_hit_at_5_lift']:.2%}",
            f"- target_85_percent_met: {summary['target_85_percent_met']}",
            "- production recommendation: no",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evidence_candidate_from_chunk(evidence_id: str, chunk: RetrievedChunk) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
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
        reasons=["phase7v_table_preview"],
    )


def _formal_citation_count(
    *,
    binder: CitationBinder,
    chunk: RetrievedChunk,
    evidence_id: str,
) -> int:
    candidate = evidence_candidate_from_chunk(evidence_id, chunk)
    support = [SupportItem(evidence_id, candidate, 0.9, ["selected_preview_table"])]
    candidates = binder.build_citation_candidates(support)
    _answer, citations, _debug = binder.bind(
        f"Preview-only table evidence [{evidence_id}].",
        support,
        citation_candidates=candidates,
    )
    return len(citations)


def _preview_metadata_ok(chunk: RetrievedChunk) -> bool:
    metadata = chunk.metadata
    return (
        metadata.get("object_type") == "table_index_unit"
        and metadata.get("table_preview") is True
        and metadata.get("index_unit_status") == "preview_only"
        and metadata.get("production_ready") is False
        and metadata.get("value_bboxes_available") is False
        and metadata.get("table_preview_allow_formal_citation") is False
        and metadata.get("citation_formal_allowed") is False
        and bool(metadata.get("table_index_unit_id"))
        and bool(metadata.get("source_csv_path"))
        and bool(metadata.get("source_pdf_crop_path"))
    )


def _expected_seen_at_20(row: dict[str, Any], provider: TablePreviewCandidateProvider) -> bool:
    expected_unit_id = row.get("expected_table_index_unit_id", "")
    candidates = provider.search(row.get("query_text", ""), top_k=20)
    candidate_ids = [candidate.chunk.metadata.get("table_index_unit_id") for candidate in candidates]
    return expected_unit_id in candidate_ids


def _non_table_controls() -> list[dict[str, str]]:
    return [
        {
            "query_text": "Summarize doc_0075 abstract and study motivation.",
            "expected_doc_id": "doc_0075",
        },
        {
            "query_text": "Explain the background objective for doc_0261 infant microbiota study.",
            "expected_doc_id": "doc_0261",
        },
        {
            "query_text": "What organism context is discussed in doc_0066 protein localization work?",
            "expected_doc_id": "doc_0066",
        },
        {
            "query_text": "Summarize doc_0076 study design and main biological system.",
            "expected_doc_id": "doc_0076",
        },
        {
            "query_text": "Explain the stated motivation for doc_0600 induction study.",
            "expected_doc_id": "doc_0600",
        },
        {
            "query_text": "What biological context is discussed in doc_0243 localization work?",
            "expected_doc_id": "doc_0243",
        },
        {
            "query_text": "Summarize the genome motif study objective in doc_0041.",
            "expected_doc_id": "doc_0041",
        },
        {
            "query_text": "Explain the enzymatic screening study background in doc_0365.",
            "expected_doc_id": "doc_0365",
        },
    ]


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return round(numerator / denominator, 6)


def _guardrails() -> dict[str, bool]:
    return {
        "production_table_index_built": False,
        "preview_units_upgraded": False,
        "formal_table_citation_generated": False,
        "canonical_source_resolution": False,
        "llm_or_ragas_called": False,
        "milvus_accessed": False,
        "official_bm25_accessed": False,
        "embedding_run": False,
        "ingestion_pipeline_modified": False,
        "route_c_implemented": False,
    }


def _ab_record_fieldnames() -> list[str]:
    return [
        "strategy",
        "query_id",
        "source_query_id",
        "query_type",
        "is_table_query",
        "query_route",
        "mode",
        "reason",
        "candidate_count",
        "merged_count",
        "preview_output_count",
        "expected_table_index_unit_id",
        "expected_unit_type",
        "shadow_expected_hit_at_20",
        "merge_expected_hit_at_5",
        "table_candidates_in_rerank_input",
        "preview_metadata_preserved",
        "formal_citation_count",
        "merged_table_index_unit_ids",
        "merged_unit_types",
    ]


def _path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase7V-fast baseline vs type-aware merge smoke.")
    parser.add_argument("--fixture-path", type=_path_arg, default=FIXTURE_PATH)
    parser.add_argument("--results-dir", type=_path_arg, default=RESULTS_DIR)
    parser.add_argument("--reports-dir", type=_path_arg, default=REPORTS_DIR)
    args = parser.parse_args()
    if not args.fixture_path.exists():
        build_ab_fixture(output_path=args.fixture_path, summary_path=args.results_dir / "ab_query_fixture_summary.json")
    summary = run_ab_smoke(
        fixture_path=args.fixture_path,
        output_dir=args.results_dir,
        reports_dir=args.reports_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
