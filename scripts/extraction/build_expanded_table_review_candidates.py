#!/usr/bin/env python3
"""Build Phase7G expanded offline table review candidates.

This script builds a larger review-only candidate pool. It reads official
chunks, Phase7D-3/Phase7E/Phase7F artifacts, and current extractor scripts as
guardrail inputs. It does not read BM25, does not access Milvus, and does not
construct gold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extraction import extract_tables_pdfplumber_v1 as pdf_extract


DEFAULT_OUTPUT_DIR = ROOT / "data/experiments/v7_phase7_expanded_table_review_pack"
DEFAULT_REPORT_DIR = ROOT / "reports/v7_phase7_expanded_table_review_pack"
DEFAULT_PDF_DIR = ROOT / "data/paper_round1/paper"
OFFICIAL_CHUNKS_PATH = ROOT / "data/baselines/phase5f_official_clean_baseline/chunks/chunks.jsonl"
OFFICIAL_CHUNKS_SHA256 = "5dbacc5bb85351203355bf3f2d22f46ec02e24f513ab9523ca3407664669f75b"

PHASE7F_INPUTS = [
    ROOT / "results/v7_phase7_gold_seed_validation/gold_seed_validation_results.json",
    ROOT / "results/v7_phase7_gold_seed_validation/gold_seed_validation_results.csv",
    ROOT / "results/v7_phase7_gold_seed_validation/formal_confirmed_validation_results.csv",
    ROOT / "results/v7_phase7_gold_seed_validation/partial_seed_exploratory_results.csv",
    ROOT / "reports/v7_phase7_gold_seed_validation/gold_seed_validation_report.md",
    ROOT / "reports/v7_phase7_gold_seed_validation/confirmed_vs_partial_seed_validation.md",
    ROOT / "reports/v7_phase7_gold_seed_validation/phase7f_summary.md",
]
PHASE7E_INPUTS = [
    ROOT / "data/experiments/v7_phase7_hybrid_gold_seed/table_gold_seed.jsonl",
    ROOT / "data/experiments/v7_phase7_hybrid_gold_seed/confirmed_seed.jsonl",
    ROOT / "data/experiments/v7_phase7_hybrid_gold_seed/partial_seed.csv",
    ROOT / "reports/v7_phase7_hybrid_gold_seed/phase7e_summary.md",
]
PHASE7D3_INPUTS = [
    ROOT / "data/experiments/v7_phase7_hybrid_extractor_v2_logical_reconstruction/table_objects.jsonl",
    ROOT / "data/experiments/v7_phase7_hybrid_extractor_v2_logical_reconstruction/table_object_routing_summary.csv",
    ROOT / "data/experiments/v7_phase7_hybrid_extractor_v2_logical_reconstruction/ready_candidate_pool.jsonl",
    ROOT / "reports/v7_phase7_hybrid_extractor_v2_logical_reconstruction/phase7d_3_summary.md",
]
CURRENT_EXTRACTOR_SCRIPTS = [
    ROOT / "scripts/extraction/extract_tables_pdfplumber_v1.py",
    ROOT / "scripts/extraction/align_chunk_pdfplumber_tables.py",
    ROOT / "scripts/extraction/build_hybrid_table_objects_v1.py",
    ROOT / "scripts/extraction/validate_hybrid_table_objects_v1.py",
    ROOT / "scripts/extraction/run_hybrid_table_extractor_v2.py",
    ROOT / "scripts/extraction/reconstruct_logical_cells_v2.py",
]

TABLE_ID_RE = re.compile(r"\b((?:Supplementary\s+)?(?:Table|TABLE)\s+[S]?\d+[A-Za-z]?)\b")
NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
UNIT_RE = re.compile(
    r"\b(?:g/L|mg/L|mmol|mol|mM|uM|%|h-1|OD\d*|CFU|kg|g\.g|mg/g|mg kg|U/mg|[0-9]+\s*[°◦]?C)\b",
    re.IGNORECASE,
)
REFERENCE_RE = re.compile(r"\b(?:Reference|References|Source|Ref\.?|this study|et\s+al\.?|DSMZ|DGCC)\b", re.IGNORECASE)
LITERAL_RE = re.compile(r"\b(?:N\.D\.|ND|N\.T\.|NT|N\.C\.|NC|LNT\s*II|LNT|2['′]-?FL|3-?FL)\b", re.IGNORECASE)
FIGURE_RE = re.compile(r"\b(?:Figure|FIGURE|Fig\.?|scheme|image|western blot|microscopy)\b", re.IGNORECASE)
MATRIX_RE = re.compile(r"\b(?:correlation matrix|matrix|heatmap|ANOVA|PCA|loading score)\b", re.IGNORECASE)
TARGET_TABLE_WORD_RE = re.compile(
    r"\b(?:strain|plasmid|primer|yield|titer|titre|activity|concentration|composition|"
    r"source|medium|condition|reference|variant|enzyme|oligosaccharide|HMO|lactose|LNT)\b",
    re.IGNORECASE,
)

REVIEW_PRIORITY_ORDER = {
    "P0_quick_review": 0,
    "P1_review": 1,
    "P2_optional_spotcheck": 2,
    "auto_excluded": 3,
}
SUGGESTED_DECISIONS = {
    "likely_confirmed_candidate",
    "likely_partial_candidate",
    "needs_rule_fix",
    "reject_boundary",
    "reject_grid",
    "backlog",
}

SCORED_FIELDS = [
    "candidate_id",
    "table_object_id",
    "doc_id",
    "table_id",
    "caption",
    "page",
    "routing_status",
    "auto_score",
    "review_priority",
    "suggested_decision",
    "reason_for_selection",
    "risk_tags",
    "markdown_path",
    "csv_path",
    "pdf_crop_path",
    "crop_status",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalize_space(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def compact_id(value: str) -> str:
    text = normalize_space(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def semicolon(values: list[Any] | tuple[Any, ...] | set[Any]) -> str:
    cleaned = [normalize_space(value) for value in values if normalize_space(value)]
    return ";".join(dict.fromkeys(cleaned)) if cleaned else "none"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_input_inventory(paths: list[Path]) -> list[dict[str, Any]]:
    inventory = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"required input missing: {rel(path)}")
        text = path.read_text(encoding="utf-8", errors="replace") if path.suffix in {".md", ".py", ".csv"} else ""
        inventory.append(
            {
                "path": rel(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "line_count": text.count("\n") + (1 if text else 0),
            }
        )
    return inventory


def load_inputs() -> dict[str, Any]:
    required = PHASE7F_INPUTS + PHASE7E_INPUTS + PHASE7D3_INPUTS + CURRENT_EXTRACTOR_SCRIPTS + [OFFICIAL_CHUNKS_PATH]
    inventory = read_input_inventory(required)
    return {
        "inventory": inventory,
        "phase7f_json": json.loads(PHASE7F_INPUTS[0].read_text(encoding="utf-8")),
        "phase7f_rows": load_csv(PHASE7F_INPUTS[1]),
        "formal_confirmed_rows": load_csv(PHASE7F_INPUTS[2]),
        "partial_exploratory_rows": load_csv(PHASE7F_INPUTS[3]),
        "table_gold_seed": load_jsonl(PHASE7E_INPUTS[0]),
        "confirmed_seed": load_jsonl(PHASE7E_INPUTS[1]),
        "partial_seed_rows": load_csv(PHASE7E_INPUTS[2]),
        "phase7d3_objects": load_jsonl(PHASE7D3_INPUTS[0]),
        "phase7d3_routing_rows": load_csv(PHASE7D3_INPUTS[1]),
        "phase7d3_ready_pool": load_jsonl(PHASE7D3_INPUTS[2]),
        "chunks_sha256": sha256_file(OFFICIAL_CHUNKS_PATH),
    }


def normalized_table_id(table_id: str) -> str:
    return compact_id(table_id.replace("supplementary", ""))


def table_id_from_text(text: str) -> str:
    match = TABLE_ID_RE.search(text or "")
    return normalize_space(match.group(1)) if match else ""


def numeric_line_count(text: str) -> int:
    lines = str(text).replace("  ", "\n").splitlines()
    count = 0
    for line in lines:
        if len(NUMERIC_RE.findall(line)) >= 2:
            count += 1
    if count:
        return count
    return max(0, len(NUMERIC_RE.findall(text)) // 4)


def source_chunk_score(chunk: dict[str, Any]) -> tuple[float, list[str], list[str], str]:
    text = normalize_space(chunk.get("text") or chunk.get("retrieval_text") or "")
    table_id = table_id_from_text(text)
    reason: list[str] = []
    tags: list[str] = []
    score = 0.0
    if chunk.get("contains_table_caption"):
        score += 0.22
        reason.append("contains_table_caption")
    if chunk.get("contains_table_text"):
        score += 0.18
        reason.append("contains_table_text")
    if table_id:
        score += 0.18
        reason.append("table_id_signal")
    if TARGET_TABLE_WORD_RE.search(text):
        score += 0.09
        reason.append("target_table_terms")
    n_lines = numeric_line_count(text)
    if n_lines >= 2:
        score += min(0.18, 0.04 * n_lines)
        reason.append("numeric_rows")
        tags.append("numeric_metric_table")
    if UNIT_RE.search(text):
        score += 0.07
        reason.append("unit_signal")
        tags.append("unit_or_footnote_table")
    if REFERENCE_RE.search(text):
        score += 0.06
        reason.append("reference_or_source_signal")
        tags.append("literal_reference_table")
    if LITERAL_RE.search(text):
        score += 0.06
        reason.append("literal_signal")
        tags.append("literal_reference_table")
    if FIGURE_RE.search(text):
        score -= 0.08
        tags.append("figure_contamination_risk")
    if MATRIX_RE.search(text):
        score -= 0.06
        tags.append("matrix_heavy")
    page_span = int(chunk.get("page_end") or 0) - int(chunk.get("page_start") or 0)
    if page_span > 1:
        score -= 0.12
        tags.append("long_table_or_cross_page")
    if len(text) > 5000:
        score -= 0.06
        tags.append("long_context_risk")
    return round(max(0.0, min(1.0, score)), 4), tags, reason, table_id


def scan_chunk_candidates(raw_target: int, existing_keys: set[tuple[str, str, int]], pdf_dir: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], dict[str, Any]] = {}
    with OFFICIAL_CHUNKS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunk = json.loads(line)
            text = normalize_space(chunk.get("text") or chunk.get("retrieval_text") or "")
            score, type_tags, reasons, table_id = source_chunk_score(chunk)
            if score < 0.25:
                continue
            if not table_id and not chunk.get("contains_table_caption") and not chunk.get("contains_table_text"):
                continue
            doc_id = chunk.get("doc_id") or ""
            page = int(chunk.get("page_start") or chunk.get("page_end") or 0)
            if not doc_id or not page:
                continue
            table_key = normalized_table_id(table_id or f"table_page_{page}")
            key = (doc_id, table_key, page)
            if key in existing_keys:
                continue
            if not pdf_extract.find_pdf(doc_id, pdf_dir):
                continue
            candidate = {
                "source_chunk_ids": [chunk.get("chunk_id", "")],
                "doc_id": doc_id,
                "table_id": table_id or "table_signal_unknown_id",
                "caption": text[:650],
                "page": page,
                "page_end": int(chunk.get("page_end") or page),
                "source_file": chunk.get("source_file") or f"{doc_id}.pdf",
                "source_text": text,
                "source_score": score,
                "source_reason_tags": reasons,
                "table_type_tags": type_tags,
                "source_kind": "official_chunk_table_signal",
            }
            previous = grouped.get(key)
            if not previous or candidate["source_score"] > previous["source_score"]:
                grouped[key] = candidate
            elif previous:
                previous["source_chunk_ids"].append(chunk.get("chunk_id", ""))

    by_doc: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for candidate in sorted(grouped.values(), key=lambda row: row["source_score"], reverse=True):
        doc_id = candidate["doc_id"]
        if by_doc[doc_id] >= 3:
            continue
        selected.append(candidate)
        by_doc[doc_id] += 1
        if len(selected) >= raw_target:
            break
    return selected


def text_overlap_score(left: str, right: str) -> float:
    def toks(value: str) -> set[str]:
        return {token.lower() for token in re.findall(r"[A-Za-z0-9]{2,}", value) if token.lower() not in {"table", "with", "from", "that", "this"}}

    ltoks = toks(left)
    rtoks = toks(right)
    if not ltoks or not rtoks:
        return 0.0
    return round(len(ltoks & rtoks) / len(ltoks | rtoks), 4)


def choose_pdf_table(candidate: dict[str, Any], pdf_dir: Path, cache: dict[tuple[str, int], list[dict[str, Any]]]) -> tuple[dict[str, Any] | None, str]:
    pdfplumber, _, import_error = pdf_extract.import_pdfplumber()
    if not pdfplumber:
        return None, f"pdfplumber_unavailable:{import_error}"
    pdf_path = pdf_extract.find_pdf(candidate["doc_id"], pdf_dir)
    if not pdf_path:
        return None, "pdf_missing"
    pages = [int(candidate.get("page") or 0)]
    page_end = int(candidate.get("page_end") or pages[0])
    if page_end and page_end != pages[0] and abs(page_end - pages[0]) <= 1:
        pages.append(page_end)
    extracted: list[dict[str, Any]] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number in pages:
                if page_number <= 0 or page_number > len(pdf.pages):
                    continue
                cache_key = (candidate["doc_id"], page_number)
                if cache_key not in cache:
                    tables, _, _ = pdf_extract.extract_page_tables(
                        pdf.pages[page_number - 1],
                        candidate["doc_id"],
                        pdf_path,
                        page_number,
                    )
                    cache[cache_key] = tables
                extracted.extend(cache[cache_key])
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not extracted:
        return None, "no_pdfplumber_table_on_candidate_page"

    caption = candidate.get("caption", "")
    table_key = normalized_table_id(candidate.get("table_id", ""))
    scored = []
    for table in extracted:
        text = normalize_space(table.get("table_text") or table.get("text_preview") or "")
        overlap = text_overlap_score(caption, text)
        table_id_match = bool(table_key and table_key == normalized_table_id(table_id_from_text(text)))
        same_page = int(table.get("page_number") or 0) == int(candidate.get("page") or 0)
        quality = {"usable": 0.35, "weak": 0.18, "likely_false_positive": -0.15, "failed": -0.25}.get(
            table.get("layout_quality_status", ""), 0.0
        )
        score = 0.26 * overlap + (0.22 if table_id_match else 0.0) + (0.12 if same_page else 0.0) + quality
        if UNIT_RE.search(text):
            score += 0.04
        if REFERENCE_RE.search(text):
            score += 0.04
        if LITERAL_RE.search(text):
            score += 0.04
        scored.append((score, table))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1], "matched_best_pdfplumber_table"


def rows_from_object(obj: dict[str, Any]) -> list[list[str]]:
    cells = obj.get("cells") or []
    if not cells:
        logical_cells = obj.get("logical_cells") or []
        columns = list(dict.fromkeys(cell.get("logical_column", "") for cell in logical_cells if cell.get("logical_column")))
        rows = list(dict.fromkeys(cell.get("row_key", "") for cell in logical_cells if cell.get("row_key")))
        if columns and rows:
            lookup = {(cell.get("row_key"), cell.get("logical_column")): normalize_space(cell.get("value_raw")) for cell in logical_cells}
            return [columns] + [[lookup.get((row_key, col), "") for col in columns] for row_key in rows]
        return []

    def index_from_cell(cell: dict[str, Any], direct_key: str, id_key: str, token: str) -> int:
        direct = cell.get(direct_key)
        if direct:
            return int(direct)
        match = re.search(rf"__{token}_(\d+)$", str(cell.get(id_key) or ""))
        return int(match.group(1)) if match else 0

    max_row = max((index_from_cell(cell, "row_index", "row_id", "row") for cell in cells), default=0)
    max_col = max((index_from_cell(cell, "column_index", "column_id", "col") for cell in cells), default=0)
    rows = [["" for _ in range(max_col)] for _ in range(max_row)]
    for cell in cells:
        row = index_from_cell(cell, "row_index", "row_id", "row")
        col = index_from_cell(cell, "column_index", "column_id", "col")
        if row and col:
            rows[row - 1][col - 1] = normalize_space(cell.get("value_raw") if "value_raw" in cell else cell.get("text"))
    return rows


def classify_table(rows: list[list[str]], text: str) -> list[str]:
    tags: list[str] = []
    numeric_rows = sum(1 for row in rows if len(NUMERIC_RE.findall(" ".join(row))) >= 2)
    if numeric_rows >= 2 or len(NUMERIC_RE.findall(text)) >= 6:
        tags.append("numeric_metric_table")
    if rows and len(rows) <= 35 and max((len(row) for row in rows), default=0) <= 9:
        tags.append("simple_row_column_table")
    if UNIT_RE.search(text):
        tags.append("unit_or_footnote_table")
    if REFERENCE_RE.search(text) or LITERAL_RE.search(text):
        tags.append("literal_reference_table")
    if not tags:
        tags.append("general_table_signal")
    return list(dict.fromkeys(tags))


def risk_tags_for_candidate(candidate: dict[str, Any], rows: list[list[str]], pdf_table: dict[str, Any] | None, base_tags: list[str]) -> list[str]:
    text = normalize_space(" ".join([" ".join(row) for row in rows]) + " " + candidate.get("caption", ""))
    risks = list(base_tags)
    row_count = len(rows)
    col_count = max((len(row) for row in rows), default=0)
    if not rows:
        risks.append("text_layer_unresolved")
    if row_count > 40 or (int(candidate.get("page_end") or candidate.get("page") or 0) - int(candidate.get("page") or 0)) > 1:
        risks.append("long_table_or_cross_page")
    if col_count > 10 or MATRIX_RE.search(text):
        risks.append("matrix_heavy")
    if FIGURE_RE.search(text):
        risks.append("figure_contamination")
    if pdf_table and pdf_table.get("layout_quality_status") == "likely_false_positive":
        risks.append("likely_false_positive")
    if pdf_table and float(pdf_table.get("empty_cell_ratio") or 0.0) >= 0.65:
        risks.append("grid_sparse_or_unreadable")
    if candidate.get("routing_status") in {"grid_rejected", "chunk_fallback", "backlog"}:
        risks.append(candidate["routing_status"])
    if candidate.get("source_kind") == "official_chunk_table_signal" and not pdf_table:
        risks.append("page_only_low_confidence")
    return list(dict.fromkeys(risks))


def score_and_decision(
    candidate: dict[str, Any],
    rows: list[list[str]],
    pdf_table: dict[str, Any] | None,
    confirmed_pattern_bonus: bool,
) -> tuple[float, str, str, list[str]]:
    text = normalize_space(" ".join([" ".join(row) for row in rows]) + " " + candidate.get("caption", ""))
    risk_tags = risk_tags_for_candidate(candidate, rows, pdf_table, candidate.get("risk_tags", []))
    score = float(candidate.get("source_score") or 0.0)
    routing_status = candidate.get("routing_status", "")
    if routing_status == "ready_for_gold_candidate":
        score += 0.22
    elif routing_status == "needs_pdfplumber_rule_fix":
        score += 0.02
    elif routing_status in {"grid_rejected", "chunk_fallback", "backlog"}:
        score -= 0.22
    if rows:
        score += 0.14
    row_count = len(rows)
    col_count = max((len(row) for row in rows), default=0)
    if 3 <= row_count <= 35 and 2 <= col_count <= 9:
        score += 0.12
    if pdf_table:
        if pdf_table.get("layout_quality_status") == "usable":
            score += 0.16
        elif pdf_table.get("layout_quality_status") == "weak":
            score += 0.04
        if pdf_table.get("cell_bboxes_available"):
            score += 0.08
    if UNIT_RE.search(text):
        score += 0.05
    if REFERENCE_RE.search(text):
        score += 0.05
    if LITERAL_RE.search(text):
        score += 0.06
    if confirmed_pattern_bonus:
        score += 0.08
    if "figure_contamination" in risk_tags:
        score -= 0.12
    if "matrix_heavy" in risk_tags:
        score -= 0.1
    if "long_table_or_cross_page" in risk_tags:
        score -= 0.12
    if "likely_false_positive" in risk_tags or "text_layer_unresolved" in risk_tags:
        score -= 0.22
    score = round(max(0.0, min(1.0, score)), 4)

    suggested = "likely_partial_candidate"
    if routing_status == "grid_rejected":
        suggested = "reject_grid"
    elif routing_status in {"chunk_fallback", "backlog"}:
        suggested = "backlog"
    elif "figure_contamination" in risk_tags and score < 0.55:
        suggested = "reject_boundary"
    elif "likely_false_positive" in risk_tags:
        suggested = "reject_grid"
    elif "text_layer_unresolved" in risk_tags:
        suggested = "backlog"
    elif routing_status == "needs_pdfplumber_rule_fix":
        suggested = "needs_rule_fix"
    elif routing_status == "ready_for_gold_candidate" and candidate.get("logical_cells_count", 0) > 0:
        suggested = "likely_confirmed_candidate"
    elif score >= 0.74 and "simple_row_column_table" in classify_table(rows, text):
        suggested = "likely_confirmed_candidate"
    return score, suggested, risk_tags, classify_table(rows, text)


def auto_exclude(candidate: dict[str, Any], score: float, suggested: str, risk_tags: list[str], rows: list[list[str]]) -> bool:
    if suggested in {"reject_grid", "reject_boundary", "backlog"} and score < 0.58:
        return True
    if "text_layer_unresolved" in risk_tags or "page_only_low_confidence" in risk_tags:
        return True
    if "likely_false_positive" in risk_tags and score < 0.72:
        return True
    if "long_table_or_cross_page" in risk_tags and score < 0.72:
        return True
    if "matrix_heavy" in risk_tags and score < 0.7:
        return True
    if not rows:
        return True
    if max((len(row) for row in rows), default=0) <= 1:
        return True
    return False


def candidate_from_phase7d3_object(
    obj: dict[str, Any],
    index: int,
    confirmed_ids: set[str],
    partial_ids: set[str],
) -> dict[str, Any]:
    rows = rows_from_object(obj)
    table_id = obj.get("table_id") or "Table"
    page = int(obj.get("page") or 0)
    table_object_id = obj.get("table_object_id") or f"{obj.get('doc_id')}__{compact_id(table_id)}__phase7g_seed"
    candidate_id = f"phase7g_candidate_{index:03d}__{obj.get('doc_id')}__{compact_id(table_id)}"
    risk_tags = list(obj.get("remaining_blockers") or obj.get("routing_blockers") or [])
    if obj.get("routing_status") in {"grid_rejected", "chunk_fallback", "backlog"}:
        risk_tags.append(obj.get("routing_status"))
    source_score = {
        "ready_for_gold_candidate": 0.62,
        "needs_pdfplumber_rule_fix": 0.48,
        "grid_rejected": 0.3,
        "chunk_fallback": 0.28,
        "backlog": 0.22,
    }.get(obj.get("routing_status"), 0.32)
    confirmed_bonus = table_object_id in confirmed_ids
    score, suggested, risks, type_tags = score_and_decision(
        {
            "source_score": source_score,
            "caption": obj.get("caption", ""),
            "routing_status": obj.get("routing_status", ""),
            "risk_tags": risk_tags,
            "logical_cells_count": len(obj.get("logical_cells") or []),
        },
        rows,
        None,
        confirmed_bonus,
    )
    if table_object_id in partial_ids and suggested == "likely_confirmed_candidate":
        suggested = "likely_partial_candidate"
    return {
        "candidate_id": candidate_id,
        "table_object_id": table_object_id,
        "source_kind": "phase7d3_v2_2_table_object",
        "doc_id": obj.get("doc_id", ""),
        "table_id": table_id,
        "caption": obj.get("caption", ""),
        "page": page,
        "page_end": page,
        "routing_status": obj.get("routing_status", ""),
        "final_action": obj.get("final_action", ""),
        "auto_score": score,
        "suggested_decision": suggested,
        "risk_tags": risks,
        "table_type_tags": type_tags,
        "reason_for_selection": semicolon(["Phase7D-3 v2.2 seed object", obj.get("routing_status", "")]),
        "source_chunk_ids": obj.get("chunk_ids") or [],
        "source_text": normalize_space(obj.get("caption", "")),
        "pdf_path": "",
        "pdf_table_bbox": obj.get("hybrid_metadata", {}).get("pdf_table_bbox"),
        "crop_bbox": obj.get("hybrid_metadata", {}).get("pdf_table_bbox"),
        "rows": rows,
        "source_span_granularity": obj.get("source_span_granularity") or "cell_level",
        "value_bboxes_available": False,
        "cell_bboxes_available": bool(obj.get("cell_bboxes_available")),
        "warnings": list(obj.get("warnings") or []) + list(obj.get("binding_warnings") or []) + list(obj.get("reconstruction_warnings") or []),
        "logical_cells_count": len(obj.get("logical_cells") or []),
        "confirmed_seed_similarity": "confirmed_seed" if table_object_id in confirmed_ids else ("partial_seed" if table_object_id in partial_ids else "none"),
    }


def candidate_from_chunk_signal(
    source: dict[str, Any],
    index: int,
    pdf_dir: Path,
    pdf_cache: dict[tuple[str, int], list[dict[str, Any]]],
    confirmed_pattern_text: str,
) -> dict[str, Any]:
    pdf_table, pdf_status = choose_pdf_table(source, pdf_dir, pdf_cache)
    rows = pdf_table.get("rows") if pdf_table else []
    text = normalize_space(" ".join([" ".join(row) for row in rows]) + " " + source.get("caption", ""))
    confirmed_bonus = text_overlap_score(text, confirmed_pattern_text) >= 0.08
    source = dict(source)
    source["routing_status"] = "phase7g_pdfplumber_review"
    risk_tags = list(source.get("table_type_tags") or [])
    source["risk_tags"] = risk_tags
    score, suggested, risks, type_tags = score_and_decision(source, rows, pdf_table, confirmed_bonus)
    table_id = source.get("table_id") or "table_signal_unknown_id"
    candidate_id = f"phase7g_candidate_{index:03d}__{source['doc_id']}__{compact_id(table_id)}__p{int(source['page']):03d}"
    pdf_path = pdf_extract.find_pdf(source["doc_id"], pdf_dir)
    bbox = pdf_table.get("bbox") if pdf_table else None
    return {
        "candidate_id": candidate_id,
        "table_object_id": f"{source['doc_id']}__{compact_id(table_id)}__phase7g_review_{index:03d}",
        "source_kind": "official_chunk_plus_pdfplumber_offline",
        "doc_id": source["doc_id"],
        "table_id": table_id,
        "caption": source.get("caption", ""),
        "page": int((pdf_table or {}).get("page_number") or source.get("page") or 0),
        "page_end": int(source.get("page_end") or source.get("page") or 0),
        "routing_status": "phase7g_pdfplumber_review",
        "final_action": "offline_review_only",
        "auto_score": score,
        "suggested_decision": suggested,
        "risk_tags": risks,
        "table_type_tags": type_tags,
        "reason_for_selection": semicolon(source.get("source_reason_tags", []) + [pdf_status]),
        "source_chunk_ids": source.get("source_chunk_ids") or [],
        "source_text": source.get("source_text", ""),
        "pdf_path": rel(pdf_path) if pdf_path else "",
        "pdf_table_bbox": bbox,
        "crop_bbox": bbox,
        "rows": rows,
        "source_span_granularity": "cell_level" if pdf_table and pdf_table.get("cell_bboxes_available") else "table_row_level",
        "value_bboxes_available": False,
        "cell_bboxes_available": bool(pdf_table and pdf_table.get("cell_bboxes_available")),
        "warnings": list((pdf_table or {}).get("extraction_warnings") or []),
        "logical_cells_count": 0,
        "confirmed_seed_similarity": "similar_pattern" if confirmed_bonus else "none",
        "pdfplumber_table_id": (pdf_table or {}).get("pdfplumber_table_id", ""),
        "pdfplumber_layout_quality_status": (pdf_table or {}).get("layout_quality_status", ""),
        "pdfplumber_extraction_confidence": (pdf_table or {}).get("extraction_confidence", ""),
    }


def assign_review_priorities(candidates: list[dict[str, Any]], review_target: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scored = []
    excluded = []
    for candidate in candidates:
        rows = candidate.get("rows") or []
        if auto_exclude(
            candidate,
            float(candidate.get("auto_score") or 0.0),
            candidate.get("suggested_decision", ""),
            list(candidate.get("risk_tags") or []),
            rows,
        ):
            candidate["review_priority"] = "auto_excluded"
            candidate["crop_status"] = "auto_excluded"
            excluded.append(candidate)
        else:
            scored.append(candidate)

    scored.sort(key=lambda row: (float(row.get("auto_score") or 0.0), row.get("cell_bboxes_available") is True), reverse=True)
    p0_target = min(24, max(18, review_target // 2))
    p1_target = min(18, max(10, review_target - p0_target - 5))
    for index, candidate in enumerate(scored):
        if index < p0_target:
            candidate["review_priority"] = "P0_quick_review"
        elif index < p0_target + p1_target:
            candidate["review_priority"] = "P1_review"
        else:
            candidate["review_priority"] = "P2_optional_spotcheck"
        candidate["crop_status"] = "pending"
    return scored + excluded, excluded


def fill_expected_paths(candidates: list[dict[str, Any]], output_dir: Path) -> None:
    for candidate in candidates:
        if candidate.get("review_priority") == "auto_excluded":
            candidate["markdown_path"] = ""
            candidate["csv_path"] = ""
            candidate["pdf_crop_path"] = ""
            continue
        cid = candidate["candidate_id"]
        candidate["markdown_path"] = rel(output_dir / "markdown_cards" / f"{cid}.md")
        candidate["csv_path"] = rel(output_dir / "csv_tables" / f"{cid}.csv")
        candidate["pdf_crop_path"] = rel(output_dir / "pdf_crops" / f"{cid}.png")


def scored_csv_row(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "table_object_id": candidate.get("table_object_id", ""),
        "doc_id": candidate.get("doc_id", ""),
        "table_id": candidate.get("table_id", ""),
        "caption": normalize_space(candidate.get("caption", ""))[:500],
        "page": candidate.get("page", ""),
        "routing_status": candidate.get("routing_status", ""),
        "auto_score": candidate.get("auto_score", ""),
        "review_priority": candidate.get("review_priority", ""),
        "suggested_decision": candidate.get("suggested_decision", ""),
        "reason_for_selection": candidate.get("reason_for_selection", ""),
        "risk_tags": semicolon(candidate.get("risk_tags") or []),
        "markdown_path": candidate.get("markdown_path", ""),
        "csv_path": candidate.get("csv_path", ""),
        "pdf_crop_path": candidate.get("pdf_crop_path", ""),
        "crop_status": candidate.get("crop_status", ""),
    }


def write_guardrail(report_dir: Path, inventory: list[dict[str, Any]]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase7G Guardrail",
        "",
        "1. 本轮定位为 expanded table review pack generation，只生成离线候选池与人工核验包。",
        "2. 本轮不是 gold construction，不构造 confirmed_seed 或 partial_seed。",
        "3. 本轮不是 validation，不运行 extractor validation、coverage evaluation 或 flat comparison。",
        "4. 本轮不是 production，不接入 ingestion 主链路、RAG、retrieval 或 benchmark。",
        "5. 本轮允许在 official chunks 上离线扩大候选池，输出仍只供人工 review。",
        "6. 本轮不读取 BM25 index，不访问或写入 Milvus。",
        "7. 本轮不运行 retrieval、embedding、rerank、Qwen、RAGAS、OCR 或 VLM。",
        "8. 本轮不伪造 value-level bbox；cell bbox 只能解释为 cell-level provenance。",
        "9. 人工 review label 需要用户后续填写，空 label 不得当作 confirmed。",
        "10. Route C 仍只是 backlog，不在本轮实施。",
        "",
        "## 已读取输入",
    ]
    for item in inventory:
        lines.append(f"- `{item['path']}`")
    (report_dir / "phase7g_guardrail.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def distribution(rows: list[dict[str, Any]], key: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = row.get(key)
        if isinstance(value, list):
            for item in value:
                counter[normalize_space(item) or "empty"] += 1
        else:
            counter[normalize_space(value) or "empty"] += 1
    return counter


def md_counter(counter: Counter[str]) -> list[str]:
    if not counter:
        return ["- 无"]
    return [f"- `{key}`：{value}" for key, value in counter.most_common()]


def write_candidate_report(candidates: list[dict[str, Any]], excluded: list[dict[str, Any]], report_dir: Path) -> None:
    review_count = sum(1 for row in candidates if row.get("review_priority") != "auto_excluded")
    doc_count = len({row.get("doc_id") for row in candidates})
    lines = [
        "# Candidate Pool Construction Report",
        "",
        "## 1. 数量",
        f"- raw candidate pool 数量：{len(candidates)}",
        f"- review pack 可选候选数量：{review_count}",
        f"- auto_excluded 数量：{len(excluded)}",
        f"- doc_id 覆盖数量：{doc_count}",
        "",
        "## 2. 表格类型分布",
        *md_counter(distribution(candidates, "table_type_tags")),
        "",
        "## 3. review_priority 分布",
        *md_counter(distribution(candidates, "review_priority")),
        "",
        "## 4. suggested_decision 分布",
        *md_counter(distribution(candidates, "suggested_decision")),
        "",
        "## 5. risk_tags 分布",
        *md_counter(distribution(candidates, "risk_tags")),
        "",
        "## 6. 为什么这些候选适合人工 review",
        "- 候选来自 official chunks 的 table/caption/numeric/source/literal 信号，并用 pdfplumber 离线抽取行列证据。",
        "- P0/P1 优先选择简单 row-column、numeric metric、unit/reference/literal 明确且 cell bbox 可用的候选。",
        "- Phase7D-3 v2.2 ready/rule-fix 产物作为 pattern seed 进入打分，但不会自动生成 gold。",
        "",
        "## 7. 为什么部分候选被 auto_excluded",
        "- 自动排除了 page_only low confidence、text layer unresolved、likely_false_positive、grid_rejected 低分、复杂 matrix、跨页长表等低价值候选。",
        "- auto_excluded 只保留在 sidecar CSV 中，不进入主 review sheet。",
    ]
    (report_dir / "candidate_pool_construction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_candidates(raw_target: int, review_target: int, output_dir: Path, report_dir: Path, pdf_dir: Path) -> list[dict[str, Any]]:
    inputs = load_inputs()
    if inputs["chunks_sha256"] != OFFICIAL_CHUNKS_SHA256:
        raise SystemExit("official chunks SHA256 drift detected; aborting Phase7G candidate construction")

    confirmed_ids = {row.get("table_object_id", "") for row in inputs["confirmed_seed"]}
    partial_ids = {row.get("table_object_id", "") for row in inputs["partial_seed_rows"]}
    confirmed_pattern_text = normalize_space(
        " ".join(
            " ".join(cell.get("value_raw", "") for cell in seed.get("gold_cells", [])[:80])
            for seed in inputs["confirmed_seed"]
        )
    )
    candidates: list[dict[str, Any]] = []
    existing_keys: set[tuple[str, str, int]] = set()
    for obj in inputs["phase7d3_objects"]:
        table_id = obj.get("table_id") or ""
        page = int(obj.get("page") or 0)
        existing_keys.add((obj.get("doc_id", ""), normalized_table_id(table_id), page))
        candidates.append(candidate_from_phase7d3_object(obj, len(candidates) + 1, confirmed_ids, partial_ids))

    remaining_target = max(0, raw_target - len(candidates))
    source_scan_target = max(remaining_target * 8, review_target * 8, 220)
    chunk_sources = scan_chunk_candidates(source_scan_target, existing_keys, pdf_dir)
    pdf_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}
    chunk_candidates: list[dict[str, Any]] = []
    for source in chunk_sources:
        chunk_candidates.append(
            candidate_from_chunk_signal(source, len(candidates) + len(chunk_candidates) + 1, pdf_dir, pdf_cache, confirmed_pattern_text)
        )

    expanded_candidates, expanded_excluded = assign_review_priorities(candidates + chunk_candidates, review_target)
    reviewable = [row for row in expanded_candidates if row.get("review_priority") != "auto_excluded"]
    excluded_ranked = sorted(expanded_excluded, key=lambda row: float(row.get("auto_score") or 0.0), reverse=True)
    reviewable.sort(
        key=lambda row: (
            REVIEW_PRIORITY_ORDER.get(row.get("review_priority", "auto_excluded"), 9),
            -float(row.get("auto_score") or 0.0),
            row.get("candidate_id", ""),
        )
    )
    reviewable_cap = min(len(reviewable), max(review_target + 15, raw_target - 15))
    candidates = (reviewable[:reviewable_cap] + excluded_ranked)[:raw_target]
    candidates, excluded = assign_review_priorities(candidates, review_target)
    candidates.sort(
        key=lambda row: (
            REVIEW_PRIORITY_ORDER.get(row.get("review_priority", "auto_excluded"), 9),
            -float(row.get("auto_score") or 0.0),
            row.get("candidate_id", ""),
        )
    )
    fill_expected_paths(candidates, output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(candidates, output_dir / "candidate_pool_raw.jsonl")
    write_csv([scored_csv_row(row) for row in candidates], output_dir / "candidate_pool_scored.csv", SCORED_FIELDS)
    write_csv([scored_csv_row(row) for row in excluded], output_dir / "auto_excluded_candidates.csv", SCORED_FIELDS)
    write_guardrail(report_dir, inputs["inventory"])
    write_candidate_report(candidates, excluded, report_dir)
    return candidates


def run(args: argparse.Namespace) -> None:
    candidates = build_candidates(args.raw_target, args.review_target, args.output_dir, args.report_dir, args.pdf_dir)
    print(
        json.dumps(
            {
                "raw_candidates": len(candidates),
                "review_candidates_available": sum(1 for row in candidates if row.get("review_priority") != "auto_excluded"),
                "auto_excluded": sum(1 for row in candidates if row.get("review_priority") == "auto_excluded"),
                "output_dir": rel(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase7G expanded table review candidates.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--raw-target", type=int, default=70)
    parser.add_argument("--review-target", type=int, default=40)
    args = parser.parse_args()
    args.output_dir = resolve_path(args.output_dir)
    args.report_dir = resolve_path(args.report_dir)
    args.pdf_dir = resolve_path(args.pdf_dir)
    return args


if __name__ == "__main__":
    run(parse_args())
