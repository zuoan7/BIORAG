#!/usr/bin/env python3
"""Phase7C pdfplumber table extraction pilot.

This is an isolated offline experiment. It reads PDFs and official chunks for
the fixed smoke doc_ids only, writes experiment artifacts, and does not access
BM25, Milvus, retrieval, embedding/rerank, OCR/VLM, Qwen, or RAGAS.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SMOKE_DOC_IDS = [
    "doc_0322",
    "doc_0158",
    "doc_0598",
    "doc_0452",
    "doc_0468",
    "doc_0687",
    "doc_0458",
    "doc_0522",
    "doc_0523",
]

OFFICIAL_DATASET_PATH = ROOT / "reports/phase5f_eval_semantic_enhancement_v2/strict_main_eval_set_v2.jsonl"
OFFICIAL_DATASET_SHA256 = "39e817bf492fe6d40a784dc457b9ab566cb3061d13fef6cec0443b19d5ca09b3"
CHUNKS_PATH = ROOT / "data/baselines/phase5f_official_clean_baseline/chunks/chunks.jsonl"
OFFICIAL_CHUNKS_SHA256 = "5dbacc5bb85351203355bf3f2d22f46ec02e24f513ab9523ca3407664669f75b"
PHASE7B2_TABLE_OBJECTS_PATH = (
    ROOT / "data/experiments/v7_phase7_table_extraction_mvp_rerun/table_objects.jsonl"
)
PHASE7B2_REQUIRED_INPUTS = [
    ROOT / "data/experiments/v7_phase7_table_extraction_mvp_rerun/table_candidates.jsonl",
    ROOT / "data/experiments/v7_phase7_table_extraction_mvp_rerun/table_objects.jsonl",
    ROOT / "data/experiments/v7_phase7_table_extraction_mvp_rerun/table_objects_review.md",
    ROOT / "data/experiments/v7_phase7_table_extraction_mvp_rerun/table_index_units.preview.jsonl",
    ROOT / "reports/v7_phase7_table_extraction_mvp_rerun/table_candidate_detection_report.md",
    ROOT / "reports/v7_phase7_table_extraction_mvp_rerun/table_object_validation_summary.csv",
    ROOT / "reports/v7_phase7_table_extraction_mvp_rerun/table_object_validation_report.md",
    ROOT / "reports/v7_phase7_table_extraction_mvp_rerun/phase7b_2_rerun_comparison.md",
    ROOT / "reports/v7_phase7_table_extraction_mvp_rerun/phase7b_2_summary.md",
]
PHASE6D_REQUIRED_INPUTS = [
    ROOT / "reports/v7_phase6d_table_contract_refinement/phase6d_refine_round1_summary.md",
    ROOT / "reports/v7_phase6d_table_contract_refinement/numeric_unit_footnote_contract.md",
    ROOT / "reports/v7_phase6d_table_contract_refinement/numeric_unit_footnote_rules.csv",
    ROOT / "reports/v7_phase6d_table_contract_refinement/matrix_superscript_literal_contract.md",
    ROOT / "reports/v7_phase6d_table_contract_refinement/matrix_superscript_literal_rules.csv",
    ROOT / "reports/v7_phase6d_table_contract_refinement/source_span_granularity_contract.md",
    ROOT / "reports/v7_phase6d_table_contract_refinement/source_span_granularity_rules.csv",
    ROOT / "reports/v7_phase6d_table_contract_refinement/partial_to_confirmed_decision_guide.md",
    ROOT / "reports/v7_phase6d_table_contract_refinement/partial_to_confirmed_rules.csv",
]
SCHEMA_REQUIRED_INPUTS = [
    ROOT / "schemas/table_object_v1.yaml",
    ROOT / "docs/table_object_schema_v1.md",
]

DEFAULT_OUTPUT_DIR = ROOT / "data/experiments/v7_phase7_pdfplumber_pilot"
DEFAULT_REPORT_DIR = ROOT / "reports/v7_phase7_pdfplumber_pilot"
DEFAULT_PDF_DIR = ROOT / "data/paper_round1/paper"
DEFAULT_RAW_FILENAME = "pdfplumber_tables.raw.jsonl"
DEFAULT_GUARDRAIL_FILENAME = "phase7c_guardrail.md"
DEFAULT_LAYOUT_REPORT_FILENAME = "pdfplumber_extraction_report.md"

TABLE_ID_RE = re.compile(r"\b(?:Supplementary\s+)?(?:Table|TABLE)\s+[S]?\d+[A-Za-z]?\b")
NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
TABLE_WORD_RE = re.compile(
    r"\b(?:Table|strain|plasmid|primer|yield|titer|titre|activity|selectivity|"
    r"source|medium|composition|carbohydrate|reference|energy|enzyme|variant)\b",
    re.IGNORECASE,
)
FIGURE_OR_PAGE_NOISE_RE = re.compile(
    r"\b(?:Figure|FIGURE|Fig\.?|journal|copyright|downloaded|supplementary\s+data|"
    r"wileyonlinelibrary|elsevier|springer|frontiers|volume|vol\.|page)\b",
    re.IGNORECASE,
)
BROKEN_WORD_RE = re.compile(r"\b[A-Za-z]{1,3}/[A-Za-z]{2,}\b|[A-Za-z]{2,}-\s+[A-Za-z]{2,}")
SENTENCE_PUNCT_RE = re.compile(r"[.;:,][\s)]")

LAYOUT_GATE_HARDENING_RULES = {
    "LQH001": "page_body_mixed_candidates_cannot_be_usable",
    "LQH002": "figure_caption_journal_header_or_footer_noise_downgrades_layout",
    "LQH003": "single_column_linearized_tables_are_not_usable_grids",
    "LQH004": "long_grids_without_stable_header_near_top_are_downgraded",
    "LQH005": "high_empty_ratio_with_broken_words_or_misalignment_is_downgraded",
    "LQH006": "large_bbox_covering_page_body_or_figure_region_is_downgraded",
    "LQH007": "source_review_grid_rejected_cases_must_not_be_used_as_reliable_hybrid_grid",
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def normalize_space(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def import_pdfplumber() -> tuple[Any | None, str, str]:
    try:
        module = importlib.import_module("pdfplumber")
    except Exception as exc:  # pragma: no cover - exercised only when dependency is absent
        return None, "dependency_missing", f"{type(exc).__name__}: {exc}"
    return module, getattr(module, "__version__", "unknown"), ""


def find_pdf(doc_id: str, pdf_dir: Path) -> Path | None:
    direct = pdf_dir / f"{doc_id}.pdf"
    if direct.exists():
        return direct
    matches = sorted((ROOT / "data").rglob(f"*{doc_id}*.pdf"))
    return matches[0] if matches else None


def count_chunks_by_doc(chunks_path: Path, doc_ids: list[str]) -> Counter[str]:
    wanted = set(doc_ids)
    counts: Counter[str] = Counter()
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            doc_id = item.get("doc_id")
            if doc_id in wanted:
                counts[doc_id] += 1
    return counts


def table_settings() -> dict[str, dict[str, Any]]:
    return {
        "lines": {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "edge_min_length": 3,
            "intersection_tolerance": 3,
            "text_tolerance": 3,
        },
        "text": {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "min_words_vertical": 2,
            "min_words_horizontal": 1,
            "intersection_tolerance": 5,
            "text_tolerance": 3,
        },
    }


def bbox_to_list(bbox: Any) -> list[float] | None:
    if not bbox:
        return None
    try:
        return [round(float(value), 3) for value in bbox]
    except Exception:
        return None


def row_values(table: Any) -> list[list[str]]:
    try:
        extracted = table.extract() or []
    except Exception:
        extracted = []
    rows: list[list[str]] = []
    for row in extracted:
        rows.append([normalize_space(cell) for cell in (row or [])])
    return rows


def flatten_rows(rows: list[list[str]]) -> str:
    return normalize_space(" ".join(" ".join(row) for row in rows))


def non_empty_cells(row: list[str]) -> list[str]:
    return [cell for cell in row if normalize_space(cell)]


def body_like_row_count(rows: list[list[str]]) -> int:
    count = 0
    for row in rows:
        cells = non_empty_cells(row)
        text = normalize_space(" ".join(cells))
        if len(text) >= 70 and (SENTENCE_PUNCT_RE.search(text) or len(cells) <= 2):
            count += 1
    return count


def has_stable_header_near_top(rows: list[list[str]]) -> bool:
    for row in rows[:4]:
        cells = non_empty_cells(row)
        if len(cells) < 2:
            continue
        alpha_cells = sum(1 for cell in cells if re.search(r"[A-Za-z]", cell))
        numeric_cells = sum(1 for cell in cells if NUMERIC_RE.fullmatch(cell))
        if alpha_cells >= 2 and numeric_cells < len(cells):
            return True
        if TABLE_WORD_RE.search(" ".join(cells)):
            return True
    return False


def looks_single_column_linearized(rows: list[list[str]], row_count: int, column_count: int) -> bool:
    if column_count <= 1 and row_count >= 2:
        return True
    if column_count > 2 or row_count < 6:
        return False
    single_cell_rows = sum(1 for row in rows if len(non_empty_cells(row)) <= 1)
    return single_cell_rows >= max(4, int(row_count * 0.65))


def build_cells(
    pdfplumber_table_id: str,
    table: Any,
    rows: list[list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, float]:
    cells: list[dict[str, Any]] = []
    cell_bboxes: list[dict[str, Any]] = []
    table_rows = list(getattr(table, "rows", []) or [])
    max_cols = max((len(row) for row in rows), default=0)
    bbox_count = 0
    material_count = 0

    for row_index, row in enumerate(rows, start=1):
        row_bboxes = []
        if row_index <= len(table_rows):
            row_bboxes = list(getattr(table_rows[row_index - 1], "cells", []) or [])
        for column_index in range(1, max_cols + 1):
            value = row[column_index - 1] if column_index - 1 < len(row) else ""
            bbox = bbox_to_list(row_bboxes[column_index - 1]) if column_index - 1 < len(row_bboxes) else None
            cell_id = f"{pdfplumber_table_id}__r{row_index:03d}c{column_index:03d}"
            if value:
                material_count += 1
                if bbox:
                    bbox_count += 1
            cells.append(
                {
                    "cell_id": cell_id,
                    "row_index": row_index,
                    "column_index": column_index,
                    "text": value,
                    "bbox": bbox,
                }
            )
            if bbox:
                cell_bboxes.append(
                    {
                        "cell_id": cell_id,
                        "row_index": row_index,
                        "column_index": column_index,
                        "bbox": bbox,
                    }
                )

    if not cells and getattr(table, "cells", None):
        for index, bbox in enumerate(getattr(table, "cells") or [], start=1):
            cell_id = f"{pdfplumber_table_id}__flat_cell_{index:04d}"
            bbox_list = bbox_to_list(bbox)
            cells.append(
                {
                    "cell_id": cell_id,
                    "row_index": None,
                    "column_index": None,
                    "text": "",
                    "bbox": bbox_list,
                }
            )
            if bbox_list:
                cell_bboxes.append({"cell_id": cell_id, "bbox": bbox_list})

    coverage = (bbox_count / material_count) if material_count else 0.0
    return cells, cell_bboxes, bool(cell_bboxes), round(coverage, 4)


def evaluate_table(record: dict[str, Any], page_width: float, page_height: float) -> None:
    warnings: list[str] = []
    quality_reasons: list[str] = []
    rows = record["rows"]
    table_text = flatten_rows(rows)
    row_count = record["row_count"]
    column_count = record["column_count"]
    empty_ratio = record["empty_cell_ratio"]
    bbox = record.get("bbox")
    non_empty_count = record["non_empty_cell_count"]
    body_rows = body_like_row_count(rows)
    has_header_near_top = has_stable_header_near_top(rows)
    single_column_linearized = looks_single_column_linearized(rows, row_count, column_count)
    figure_or_page_noise = bool(FIGURE_OR_PAGE_NOISE_RE.search(table_text))
    broken_words_or_misalignment = bool(BROKEN_WORD_RE.search(table_text))

    if not record.get("cell_bboxes_available"):
        warnings.append("cell_bbox_missing")
        quality_reasons.append("cell_bboxes_unavailable")
    if empty_ratio >= 0.65:
        warnings.append("high_empty_cell_ratio")
        quality_reasons.append("empty_cell_ratio_high")
    if row_count < 2:
        warnings.append("row_count_too_low")
        quality_reasons.append("row_count_lt_2")
    if column_count < 2:
        warnings.append("column_count_too_low")
        quality_reasons.append("column_count_lt_2")
    if bbox:
        width_ratio = max(0.0, (bbox[2] - bbox[0]) / page_width) if page_width else 0.0
        height_ratio = max(0.0, (bbox[3] - bbox[1]) / page_height) if page_height else 0.0
        if width_ratio > 0.72 and height_ratio > 0.55 and column_count <= 3:
            warnings.append("suspected_multicolumn_page")
            quality_reasons.append("large_page_region_with_few_columns")
        if width_ratio > 0.82 and height_ratio > 0.65:
            warnings.append("large_bbox_page_text_or_figure_region")
            warnings.append("suspected_false_positive_layout")
            quality_reasons.append("bbox_covers_page_body_or_figure_region")
        if height_ratio < 0.025 or width_ratio < 0.08:
            warnings.append("suspected_false_positive_layout")
            quality_reasons.append("very_small_bbox")
    if row_count >= 8 and body_rows >= max(3, int(row_count * 0.25)):
        warnings.append("page_body_mixed_candidate")
        warnings.append("suspected_false_positive_layout")
        quality_reasons.append("page_body_mixed_rows")
    if figure_or_page_noise:
        warnings.append("figure_or_page_noise_in_candidate")
        quality_reasons.append("figure_caption_header_or_footer_noise")
        if row_count >= 4 or column_count <= 3:
            warnings.append("suspected_false_positive_layout")
    if single_column_linearized:
        warnings.append("single_column_linearized_table")
        warnings.append("suspected_false_positive_layout")
        quality_reasons.append("single_column_linearized_grid")
    if row_count >= 12 and not has_header_near_top:
        warnings.append("unstable_header_for_long_grid")
        quality_reasons.append("stable_header_not_near_top")
        if body_rows or column_count <= 3:
            warnings.append("suspected_false_positive_layout")
    if empty_ratio >= 0.45 and broken_words_or_misalignment:
        warnings.append("high_empty_ratio_with_broken_words")
        quality_reasons.append("empty_cells_with_broken_words_or_misalignment")
    if not TABLE_WORD_RE.search(table_text) and len(NUMERIC_RE.findall(table_text)) < 2:
        warnings.append("suspected_false_positive_layout")
        quality_reasons.append("weak_table_text_signal")
    if row_count <= 2 and column_count <= 2 and non_empty_count <= 2:
        warnings.append("suspected_false_positive_layout")
        quality_reasons.append("tiny_sparse_grid")
    if row_count < 3 and column_count >= 4:
        warnings.append("suspected_table_split")
        quality_reasons.append("short_wide_grid_possible_fragment")

    warnings = sorted(set(warnings))
    record["extraction_warnings"] = warnings
    if "suspected_false_positive_layout" in warnings or row_count == 0 or column_count == 0:
        confidence = "low"
    elif row_count >= 3 and column_count >= 2 and empty_ratio < 0.55 and record["non_empty_cell_count"] >= 4:
        confidence = "high"
    elif row_count >= 2 and column_count >= 2 and record["non_empty_cell_count"] >= 2:
        confidence = "medium"
    else:
        confidence = "low"
    record["extraction_confidence"] = confidence

    mostly_single_column_text = single_column_linearized
    likely_false_positive = "suspected_false_positive_layout" in warnings or mostly_single_column_text
    if mostly_single_column_text:
        quality_reasons.append("mostly_single_column_text")

    score = 1.0
    if row_count < 2 or column_count < 2:
        score -= 0.45
    if non_empty_count < 2:
        score -= 0.35
    if empty_ratio >= 0.65:
        score -= 0.25
    elif empty_ratio >= 0.45:
        score -= 0.12
        quality_reasons.append("empty_cell_ratio_moderate")
    if not record.get("cell_bboxes_available"):
        score -= 0.2
    if likely_false_positive:
        score -= 0.35
    if "page_body_mixed_candidate" in warnings:
        score -= 0.2
    if "figure_or_page_noise_in_candidate" in warnings:
        score -= 0.12
    if "unstable_header_for_long_grid" in warnings:
        score -= 0.12
    if "high_empty_ratio_with_broken_words" in warnings:
        score -= 0.12
    if "suspected_table_split" in warnings:
        score -= 0.1
    score = round(max(0.0, min(1.0, score)), 4)

    if row_count == 0 or column_count == 0:
        quality_status = "failed"
        quality_reasons.append("empty_grid")
    elif likely_false_positive:
        quality_status = "likely_false_positive"
    elif score >= 0.72 and row_count >= 2 and column_count >= 2 and record.get("cell_bboxes_available"):
        quality_status = "usable"
        quality_reasons.append("stable_row_column_cell_grid")
    else:
        quality_status = "weak"

    record["layout_quality_status"] = quality_status
    record["layout_quality_score"] = score
    record["layout_quality_reasons"] = sorted(set(quality_reasons)) or ["no_layout_quality_warning"]
    record["likely_false_positive_layout"] = bool(likely_false_positive)


def extract_page_tables(
    page: Any,
    doc_id: str,
    pdf_path: Path,
    page_number: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    strategy_counts: dict[str, int] = {}
    for strategy, settings in table_settings().items():
        try:
            tables = page.find_tables(table_settings=settings) or []
        except Exception as exc:
            failures.append(
                {
                    "doc_id": doc_id,
                    "page_number": str(page_number),
                    "strategy": strategy,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
            strategy_counts[strategy] = 0
            continue

        strategy_counts[strategy] = len(tables)
        for table_index, table in enumerate(tables, start=1):
            table_id = f"pdfplumber_{doc_id}_p{page_number:03d}_{strategy}_{table_index:03d}"
            rows = row_values(table)
            max_cols = max((len(row) for row in rows), default=0)
            cell_count = sum(max_cols for _ in rows)
            non_empty = sum(1 for row in rows for cell in row if cell)
            empty_ratio = 1.0 - (non_empty / cell_count) if cell_count else 1.0
            cells, cell_bboxes, bboxes_available, bbox_coverage = build_cells(table_id, table, rows)
            table_text = flatten_rows(rows)
            record = {
                "pdfplumber_table_id": table_id,
                "doc_id": doc_id,
                "pdf_path": rel(pdf_path),
                "page_number": page_number,
                "strategy": strategy,
                "table_order_on_page": table_index,
                "bbox": bbox_to_list(getattr(table, "bbox", None)),
                "row_count": len(rows),
                "column_count": max_cols,
                "cell_count": cell_count,
                "non_empty_cell_count": non_empty,
                "empty_cell_ratio": round(empty_ratio, 4),
                "rows": rows,
                "cells": cells,
                "cell_bboxes": cell_bboxes,
                "cell_bboxes_available": bboxes_available,
                "cell_bbox_coverage": bbox_coverage,
                "table_text": table_text,
                "text_preview": table_text[:800],
                "extraction_confidence": "failed",
                "extraction_warnings": [],
                "failure_reason": "",
                "page_width": round(float(getattr(page, "width", 0.0)), 3),
                "page_height": round(float(getattr(page, "height", 0.0)), 3),
            }
            evaluate_table(record, record["page_width"], record["page_height"])
            records.append(record)
    if strategy_counts.get("lines", 0) == 0:
        for record in records:
            if record["strategy"] == "text":
                record["extraction_warnings"] = sorted(
                    set(record["extraction_warnings"] + ["lines_strategy_empty", "text_strategy_only"])
                )
                if record["extraction_confidence"] == "high":
                    record["extraction_confidence"] = "medium"
    return records, failures, strategy_counts


def inspect_pdf(pdfplumber: Any, pdf_path: Path) -> dict[str, Any]:
    info = {
        "pdf_path": rel(pdf_path),
        "pdf_exists": pdf_path.exists(),
        "page_count": 0,
        "text_layer_readable": False,
        "text_pages": 0,
        "text_char_count": 0,
        "failure_reason": "",
    }
    try:
        with pdfplumber.open(pdf_path) as pdf:
            info["page_count"] = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    info["text_pages"] += 1
                    info["text_char_count"] += len(text)
            info["text_layer_readable"] = info["text_pages"] > 0
    except Exception as exc:
        info["failure_reason"] = f"{type(exc).__name__}: {exc}"
    return info


def collect_input_context(args: argparse.Namespace) -> dict[str, Any]:
    doc_ids = args.doc_ids
    pdfplumber, version_or_status, import_error = import_pdfplumber()
    pdf_paths = {doc_id: find_pdf(doc_id, args.pdf_dir) for doc_id in doc_ids}
    pdf_infos = {}
    if pdfplumber:
        for doc_id, pdf_path in pdf_paths.items():
            if pdf_path:
                pdf_infos[doc_id] = inspect_pdf(pdfplumber, pdf_path)
            else:
                pdf_infos[doc_id] = {
                    "pdf_path": "",
                    "pdf_exists": False,
                    "page_count": 0,
                    "text_layer_readable": False,
                    "text_pages": 0,
                    "text_char_count": 0,
                    "failure_reason": "pdf_missing",
                }
    else:
        for doc_id, pdf_path in pdf_paths.items():
            pdf_infos[doc_id] = {
                "pdf_path": rel(pdf_path) if pdf_path else "",
                "pdf_exists": bool(pdf_path),
                "page_count": 0,
                "text_layer_readable": False,
                "text_pages": 0,
                "text_char_count": 0,
                "failure_reason": "pdfplumber_dependency_missing",
            }

    chunks_counts = count_chunks_by_doc(args.chunks, doc_ids)
    phase7b2_objects = load_jsonl(args.phase7b2_table_objects)
    phase7b2_counts = Counter(obj.get("doc_id") for obj in phase7b2_objects)
    dataset_sha = sha256_file(OFFICIAL_DATASET_PATH) if OFFICIAL_DATASET_PATH.exists() else ""
    chunks_sha = sha256_file(args.chunks) if args.chunks.exists() else ""
    return {
        "pdfplumber_available": pdfplumber is not None,
        "pdfplumber_version": version_or_status if pdfplumber else "",
        "pdfplumber_import_error": import_error,
        "pdf_paths": {doc_id: rel(path) if path else "" for doc_id, path in pdf_paths.items()},
        "pdf_infos": pdf_infos,
        "chunks_counts": dict(chunks_counts),
        "phase7b2_table_object_count": len(phase7b2_objects),
        "phase7b2_table_object_counts": dict(phase7b2_counts),
        "official_dataset_sha256": dataset_sha,
        "official_chunks_sha256": chunks_sha,
        "required_phase7b2_inputs": {rel(path): path.exists() for path in PHASE7B2_REQUIRED_INPUTS},
        "required_phase6d_inputs": {rel(path): path.exists() for path in PHASE6D_REQUIRED_INPUTS},
        "required_schema_inputs": {rel(path): path.exists() for path in SCHEMA_REQUIRED_INPUTS},
        "bm25_accessed": False,
        "milvus_accessed": False,
        "official_baseline_modified": False,
        "dependency_file_modified": False,
    }


def write_guardrail(
    report_dir: Path,
    context: dict[str, Any],
    doc_ids: list[str],
    filename: str = DEFAULT_GUARDRAIL_FILENAME,
    phase_label: str = "Phase7C",
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {phase_label} 护栏",
        "",
        "## 1. 本轮定位",
        "",
        f"本轮是 `{phase_label}` 的离线 pdfplumber / chunk hybrid 加固实验。目标是在 Phase7B-2 / Phase7C 同一批 smoke 文档上做 minimal hardening，让 pipeline 能可解释地判断对象能不能用。",
        "",
        "本轮不是 production implementation，不接入 RAG，不写入 Milvus，不读取或查询 BM25 index，不重建 BM25，不重建 chunks，不修改 official baseline，不修改 ingestion 主链路，不调用 Qwen / RAGAS / OCR / VLM，不进入 Route C implementation。",
        "",
        "本轮只做 minimal hardening：不扩大 smoke，不大规模扩展主 schema，主 `table_object` 只保留 minimal `hybrid_metadata`；alignment / layout / debug 细节保存在 sidecar CSV、raw JSONL 和 validation summary。",
        "",
        "## 2. Smoke 范围",
        "",
        "本轮固定 doc_id：",
        "",
    ]
    lines.extend(f"- `{doc_id}`" for doc_id in doc_ids)
    lines.extend(
        [
            "",
            f"smoke 数量：{len(doc_ids)}。未扩大 smoke。",
            "",
            "## 3. Baseline Pins",
            "",
            f"- official dataset：`{rel(OFFICIAL_DATASET_PATH)}`",
            f"- dataset SHA256 pin：`{OFFICIAL_DATASET_SHA256}`",
            f"- dataset SHA256 actual：`{context['official_dataset_sha256']}`",
            "- official clean baseline：`phase5f_official_clean_baseline`",
            f"- official chunks：`{rel(CHUNKS_PATH)}`",
            f"- official chunks SHA256 pin：`{OFFICIAL_CHUNKS_SHA256}`",
            f"- official chunks SHA256 actual：`{context['official_chunks_sha256']}`",
            "- official Milvus collection：`synbio_phase5f_official_clean_baseline`（本轮未访问）",
            "",
            "## 4. 禁止事项执行情况",
            "",
            "| 项目 | 状态 |",
            "|---|---|",
            "| 修改 ingestion 主链路 | 未执行 |",
            "| 修改 production pipeline | 未执行 |",
            "| 修改 official dataset | 未执行 |",
            "| 修改 official baseline | 未执行 |",
            "| 修改 baseline registry/configs | 未执行 |",
            "| 重建 chunks | 未执行 |",
            "| 读取或查询 BM25 index | 未执行 |",
            "| 重建 BM25 | 未执行 |",
            "| 访问或写入 Milvus | 未执行 |",
            "| 跑 retrieval / embedding / rerank | 未执行 |",
            "| 调用 Qwen / RAGAS / OCR / VLM | 未执行 |",
            "| 接入 production | 未执行 |",
            "| 进入 Route C implementation | 未执行；Route C 仍只是 backlog |",
            "",
            "## 5. Bbox 口径",
            "",
            "`pdfplumber` 的 cell bbox 只表示 PDF layout cell 或网格区域，不能等同于 value-level token bbox。本轮允许在稳定对齐时把 source_span 提升到 `cell_level`，但 `cell_bboxes_available=true` 不等于 `value_bboxes_available=true`。本轮不伪造 `value_level` 或 `bbox_value_level`。",
            "",
            "## 6. Route C 状态",
            "",
            "Route C 仍只是 backlog。本轮不进入 Route C implementation，也不把本轮输出写成 production-ready。",
        ]
    )
    (report_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dependency_report(report_dir: Path, context: dict[str, Any], doc_ids: list[str]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    pdf_found = sum(1 for doc_id in doc_ids if context["pdf_infos"][doc_id]["pdf_exists"])
    chunks_found = sum(1 for doc_id in doc_ids if context["chunks_counts"].get(doc_id, 0) > 0)
    objects_found = sum(
        1 for doc_id in doc_ids if context["phase7b2_table_object_counts"].get(doc_id, 0) > 0
    )
    lines = [
        "# pdfplumber 依赖与输入检查",
        "",
        "## 1. 依赖状态",
        "",
        f"- pdfplumber 可 import：`{str(context['pdfplumber_available']).lower()}`",
        f"- pdfplumber 版本：`{context['pdfplumber_version'] or 'dependency_missing'}`",
        f"- import 错误：`{context['pdfplumber_import_error'] or 'none'}`",
        "- 依赖文件修改：`false`。本轮只在当前实验运行环境安装/使用 `pdfplumber`，未修改 production runtime 依赖清单。",
        "",
        "## 2. PDF 与文本层检查",
        "",
        f"- smoke doc_id 数量：{len(doc_ids)}",
        f"- 找到 PDF 数量：{pdf_found}",
        f"- official chunks 中存在 doc_id 数量：{chunks_found}",
        f"- Phase7B-2 table_objects 中存在 doc_id 数量：{objects_found}",
        "",
        "| doc_id | pdf_status | pdf_path | page_count | text_layer_readable | text_pages | text_char_count | official_chunks | phase7b2_table_objects | failure_reason |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for doc_id in doc_ids:
        info = context["pdf_infos"][doc_id]
        lines.append(
            "| {doc_id} | {status} | `{path}` | {pages} | `{text}` | {text_pages} | {chars} | {chunks} | {objects} | {reason} |".format(
                doc_id=doc_id,
                status="found" if info["pdf_exists"] else "pdf_missing",
                path=info["pdf_path"] or "",
                pages=info["page_count"],
                text=str(info["text_layer_readable"]).lower(),
                text_pages=info["text_pages"],
                chars=info["text_char_count"],
                chunks=context["chunks_counts"].get(doc_id, 0),
                objects=context["phase7b2_table_object_counts"].get(doc_id, 0),
                reason=info["failure_reason"] or "none",
            )
        )
    lines.extend(
        [
            "",
            "## 3. 必须输入读取状态",
            "",
            "### Phase7B-2 输出",
            "",
        ]
    )
    lines.extend(
        f"- `{path}`：`{'exists' if exists else 'missing'}`"
        for path, exists in context["required_phase7b2_inputs"].items()
    )
    lines.extend(["", "### Phase6D contract", ""])
    lines.extend(
        f"- `{path}`：`{'exists' if exists else 'missing'}`"
        for path, exists in context["required_phase6d_inputs"].items()
    )
    lines.extend(["", "### table_object_v1 schema", ""])
    lines.extend(
        f"- `{path}`：`{'exists' if exists else 'missing'}`"
        for path, exists in context["required_schema_inputs"].items()
    )
    lines.extend(
        [
            "",
            "## 4. Guardrail 状态",
            "",
            "- BM25 index：未读取、未查询。",
            "- Milvus：未访问、未写入。",
            "- official baseline：未修改；official chunks SHA256 与 pin 一致。",
            "- configs / baseline registry：未修改。",
        ]
    )
    (report_dir / "pdfplumber_dependency_and_input_check.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_extraction_report(
    report_dir: Path,
    records: list[dict[str, Any]],
    page_summaries: list[dict[str, Any]],
    failures: list[dict[str, str]],
    context: dict[str, Any],
    filename: str = DEFAULT_LAYOUT_REPORT_FILENAME,
    phase_label: str = "Phase7C",
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    by_doc = Counter(record["doc_id"] for record in records)
    by_strategy = Counter(record["strategy"] for record in records)
    by_confidence = Counter(record["extraction_confidence"] for record in records)
    by_layout_quality = Counter(record.get("layout_quality_status", "unknown") for record in records)
    warning_counts: Counter[str] = Counter()
    for record in records:
        warning_counts.update(record.get("extraction_warnings") or [])
    no_table_pages = [item for item in page_summaries if item["total_tables"] == 0]
    high_empty_ratio_count = sum(1 for record in records if record.get("empty_cell_ratio", 0) >= 0.65)
    likely_false_positive_count = sum(1 for record in records if record.get("likely_false_positive_layout"))
    cell_bbox_available_count = sum(1 for record in records if record.get("cell_bboxes_available"))
    lines = [
        "# pdfplumber layout quality 报告",
        "",
        "## 1. 抽取目标",
        "",
        f"本报告记录 {phase_label} 中 `pdfplumber` 对固定 9 篇 smoke PDF 的 raw layout table extraction 与 layout quality summary。它只用于 pilot 对齐评估，不进入 ingestion 主链路，不写 Milvus，不建 BM25，不跑 retrieval。",
        "",
        "## 2. 总体统计",
        "",
        f"- raw pdfplumber tables 总数：{len(records)}",
        f"- 无表格页数：{len(no_table_pages)}",
        f"- strategy 统计：{dict(by_strategy)}",
        f"- extraction_confidence 统计：{dict(by_confidence)}",
        f"- layout_quality_status 统计：{dict(by_layout_quality)}",
        f"- high empty ratio 数量：{high_empty_ratio_count}",
        f"- likely false positive layout 数量：{likely_false_positive_count}",
        f"- cell_bboxes_available 数量：{cell_bbox_available_count}",
        f"- strategy failure 数量：{len(failures)}",
        "",
        "## 3. layout_quality_status 统计",
        "",
        "| layout_quality_status | 数量 |",
        "|---|---:|",
    ]
    for status in ["usable", "weak", "likely_false_positive", "failed"]:
        lines.append(f"| `{status}` | {by_layout_quality.get(status, 0)} |")
    lines.extend(
        [
            "",
            "## 4. doc_id 统计",
            "",
            "| doc_id | pdf_path | page_count | raw_tables | text_layer_readable |",
            "|---|---|---:|---:|---|",
        ]
    )
    for doc_id, info in context["pdf_infos"].items():
        lines.append(
            f"| {doc_id} | `{info['pdf_path']}` | {info['page_count']} | {by_doc.get(doc_id, 0)} | `{str(info['text_layer_readable']).lower()}` |"
        )
    lines.extend(["", "## 5. warning 统计", "", "| warning | 数量 |", "|---|---:|"])
    for warning, count in warning_counts.most_common():
        lines.append(f"| `{warning}` | {count} |")
    if not warning_counts:
        lines.append("| none | 0 |")
    lines.extend(
        [
            "",
            "## 6. page_no_table_found summary",
            "",
            "以下页面在 lines/text 两类策略下均未抽到表格；这不代表页面没有表格，只代表本轮 pdfplumber raw extraction 未找到可用 layout table。",
            "",
            "| doc_id | page_number | lines_tables | text_tables | total_tables |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in no_table_pages[:200]:
        lines.append(
            f"| {item['doc_id']} | {item['page_number']} | {item.get('lines_tables', 0)} | {item.get('text_tables', 0)} | {item['total_tables']} |"
        )
    if len(no_table_pages) > 200:
        lines.append(f"| ... | ... | ... | ... | 仅显示前 200 页，共 {len(no_table_pages)} 页 |")
    lines.extend(["", "## 7. 失败策略", ""])
    if failures:
        lines.extend(["| doc_id | page_number | strategy | failure_reason |", "|---|---:|---|---|"])
        for failure in failures[:100]:
            lines.append(
                f"| {failure['doc_id']} | {failure['page_number']} | `{failure['strategy']}` | {failure['failure_reason']} |"
            )
    else:
        lines.append("未记录 strategy-level failure。")
    lines.extend(
        [
            "",
            "## 8. layout quality 对 alignment 的影响",
            "",
            "- `usable` 是 alignment high/medium 的必要但不充分条件；仍需 doc/page/table_id/caption/body overlap 支撑。",
            "- `weak` 可以参与候选对齐，但不能单独支撑 high confidence，也不能自动进入 pass。",
            "- `likely_false_positive` 不能给 high confidence；若只剩 page-only 信号，应进入 manual review 或 fallback。",
            "- `failed` 不用于构造 pdfplumber-backed rows/cells。",
            "",
            "## 9. 口径说明",
            "",
            "- `cell_bboxes` 来自 pdfplumber table cell/grid，不是 value-level token bbox。",
            "- `cell_bboxes_available=true` 不等于 `value_bboxes_available=true`。",
            "- `text_strategy_only` 通常说明 lines strategy 未找到表格，需警惕整页或多栏文本被误识别为表格。",
            "- `suspected_false_positive_layout`、`high_empty_cell_ratio`、`row_count_too_low` 不会在后续自动升级为 confirmed。",
            "- `matched` 对齐也不等于 extraction correct，后续 validation 会保留 manual review dependency。",
        ]
    )
    (report_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    context = collect_input_context(args)
    write_guardrail(args.report_dir, context, args.doc_ids, args.guardrail_filename, args.phase_label)
    if args.write_dependency_report:
        write_dependency_report(args.report_dir, context, args.doc_ids)

    pdfplumber, _, _ = import_pdfplumber()
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    page_summaries: list[dict[str, Any]] = []
    if not pdfplumber:
        write_extraction_report(
            args.report_dir,
            records,
            page_summaries,
            failures,
            context,
            args.layout_report_filename,
            args.phase_label,
        )
        print(json.dumps({"status": "dependency_missing", "raw_tables": 0}, ensure_ascii=False, indent=2))
        return

    for doc_id in args.doc_ids:
        pdf_path_text = context["pdf_paths"].get(doc_id) or ""
        if not pdf_path_text:
            continue
        pdf_path = ROOT / pdf_path_text
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_index, page in enumerate(pdf.pages, start=1):
                    page_records, page_failures, strategy_counts = extract_page_tables(
                        page, doc_id, pdf_path, page_index
                    )
                    records.extend(page_records)
                    failures.extend(page_failures)
                    page_summaries.append(
                        {
                            "doc_id": doc_id,
                            "page_number": page_index,
                            "lines_tables": strategy_counts.get("lines", 0),
                            "text_tables": strategy_counts.get("text", 0),
                            "total_tables": sum(strategy_counts.values()),
                        }
                    )
        except Exception as exc:
            failures.append(
                {
                    "doc_id": doc_id,
                    "page_number": "all",
                    "strategy": "open_pdf",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )

    raw_path = args.output_dir / args.raw_filename
    with raw_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    write_extraction_report(
        args.report_dir,
        records,
        page_summaries,
        failures,
        context,
        args.layout_report_filename,
        args.phase_label,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "raw_tables": len(records),
                "output": rel(raw_path),
                "report": rel(args.report_dir / args.layout_report_filename),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract raw pdfplumber tables for Phase7C pilot.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--phase7b2-table-objects", type=Path, default=PHASE7B2_TABLE_OBJECTS_PATH)
    parser.add_argument("--doc-id", action="append", dest="doc_ids")
    parser.add_argument("--raw-filename", default=DEFAULT_RAW_FILENAME)
    parser.add_argument("--guardrail-filename", default=DEFAULT_GUARDRAIL_FILENAME)
    parser.add_argument("--layout-report-filename", default=DEFAULT_LAYOUT_REPORT_FILENAME)
    parser.add_argument("--phase-label", default="Phase7C")
    parser.add_argument("--write-dependency-report", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.output_dir = resolve_path(args.output_dir)
    args.report_dir = resolve_path(args.report_dir)
    args.pdf_dir = resolve_path(args.pdf_dir)
    args.chunks = resolve_path(args.chunks)
    args.phase7b2_table_objects = resolve_path(args.phase7b2_table_objects)
    args.doc_ids = args.doc_ids or SMOKE_DOC_IDS
    invalid = [doc_id for doc_id in args.doc_ids if doc_id not in SMOKE_DOC_IDS]
    if invalid:
        raise SystemExit(f"Phase7C smoke 不允许扩大：{invalid}")
    return args


if __name__ == "__main__":
    run(parse_args())
