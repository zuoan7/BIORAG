#!/usr/bin/env python3
"""Align Phase7B-2 chunk table_objects with pdfplumber raw tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
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
DEFAULT_TABLE_OBJECTS_PATH = (
    ROOT / "data/experiments/v7_phase7_table_extraction_mvp_rerun/table_objects.jsonl"
)
DEFAULT_PDFPLUMBER_RAW_PATH = (
    ROOT / "data/experiments/v7_phase7_pdfplumber_pilot/pdfplumber_tables.raw.jsonl"
)
DEFAULT_CHUNKS_PATH = ROOT / "data/baselines/phase5f_official_clean_baseline/chunks/chunks.jsonl"
DEFAULT_OUTPUT_PATH = (
    ROOT / "data/experiments/v7_phase7_pdfplumber_pilot/chunk_pdfplumber_alignment.csv"
)
DEFAULT_REPORT_PATH = (
    ROOT / "reports/v7_phase7_pdfplumber_pilot/chunk_pdfplumber_alignment_report.md"
)
DEFAULT_PHASE7C_ALIGNMENT_PATH = (
    ROOT / "data/experiments/v7_phase7_pdfplumber_pilot/chunk_pdfplumber_alignment.csv"
)
DEFAULT_PDF_DIR = ROOT / "data/paper_round1/paper"

CSV_FIELDS = [
    "chunk_table_object_id",
    "doc_id",
    "table_id",
    "chunk_page",
    "chunk_validation_status",
    "pdfplumber_table_id",
    "pdf_page",
    "pdf_strategy",
    "pdf_table_bbox",
    "layout_quality_status",
    "alignment_status",
    "alignment_confidence",
    "alignment_score",
    "alignment_basis",
    "alignment_blockers",
    "needs_manual_alignment_review",
    "notes",
]

TABLE_ID_RE = re.compile(r"\b((?:Supplementary\s+)?(?:Table|TABLE)\s+[S]?\d+[A-Za-z]?)\b")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "table",
    "used",
    "study",
    "data",
    "are",
    "were",
    "was",
    "into",
    "using",
    "source",
    "reference",
    "caption",
}

ALIGNMENT_GATE_HARDENING_RULES = {
    "AGH001": "page_only_match_is_always_low_confidence_manual_review",
    "AGH002": "same_table_id_requires_caption_title_or_grid_proximity_not_body_reference_only",
    "AGH003": "same_page_multiple_candidates_require_layout_bbox_caption_body_and_overlap_signals",
    "AGH004": "conflict_and_multiple_pdf_tables_must_enter_manual_review_or_reject",
    "AGH005": "source_review_rejected_pdfplumber_candidates_cannot_remain_usable_hybrid",
    "AGH006": "alignment_confirmed_grid_rejected_keeps_alignment_evidence_but_rejects_cell_grid",
    "AGH007": "keep_hybrid_candidate_cases_enter_binding_review_queue_only",
}


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_table_id(table_id: str) -> str:
    text = normalize_space(table_id).lower()
    text = re.sub(r"\bcontinued\b", "", text)
    text = text.replace("supplementary", "")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def table_id_from_text(text: str) -> str:
    match = TABLE_ID_RE.search(text or "")
    return normalize_space(match.group(1)) if match else ""


def table_id_near_candidate_start(table_id: str, text: str, limit: int = 650) -> bool:
    """Return whether the table id appears near title/grid start, not only body text."""

    table_key = normalize_table_id(table_id)
    if not table_key:
        return False
    early_text = normalize_space(text)[:limit]
    for match in TABLE_ID_RE.finditer(early_text):
        if normalize_table_id(match.group(1)) == table_key:
            return True
    return False


def tokens(text: str) -> set[str]:
    result = set()
    for token in re.findall(r"[A-Za-z0-9]+", normalize_space(text).lower()):
        if len(token) < 2 or token in STOPWORDS:
            continue
        result.add(token)
    return result


def caption_text_overlap_score(left: str, right: str) -> float:
    left_tokens = tokens(left)
    right_tokens = tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return round(len(left_tokens & right_tokens) / len(left_tokens | right_tokens), 4)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def load_chunks_by_id(path: Path, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
    wanted = set(doc_ids)
    chunks: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("doc_id") in wanted:
                chunks[item.get("chunk_id", "")] = item
    return chunks


def find_pdf(doc_id: str, pdf_dir: Path) -> Path | None:
    direct = pdf_dir / f"{doc_id}.pdf"
    if direct.exists():
        return direct
    matches = sorted((ROOT / "data").rglob(f"*{doc_id}*.pdf"))
    return matches[0] if matches else None


def chunk_nearby_text(obj: dict[str, Any], chunks_by_id: dict[str, dict[str, Any]]) -> str:
    values = [obj.get("caption", "")]
    for chunk_id in obj.get("chunk_ids") or []:
        chunk = chunks_by_id.get(chunk_id)
        if chunk:
            values.append(chunk.get("text", ""))
    return normalize_space(" ".join(values))[:4000]


def chunk_body_text(obj: dict[str, Any], chunks_by_id: dict[str, dict[str, Any]]) -> str:
    values: list[str] = []
    for chunk_id in obj.get("chunk_ids") or []:
        chunk = chunks_by_id.get(chunk_id)
        if chunk:
            values.append(chunk.get("text", ""))
    return normalize_space(" ".join(values))[:4000]


def layout_status(pdf: dict[str, Any]) -> str:
    status = pdf.get("layout_quality_status")
    if status:
        return status
    warnings = set(pdf.get("extraction_warnings") or [])
    if "suspected_false_positive_layout" in warnings:
        return "likely_false_positive"
    if pdf.get("row_count", 0) == 0 or pdf.get("column_count", 0) == 0:
        return "failed"
    if pdf.get("extraction_confidence") == "high" and pdf.get("cell_bboxes_available"):
        return "usable"
    return "weak"


def layout_score(pdf: dict[str, Any]) -> float:
    if pdf.get("layout_quality_score") is not None:
        return float(pdf.get("layout_quality_score") or 0.0)
    status = layout_status(pdf)
    return {"usable": 0.9, "weak": 0.45, "likely_false_positive": 0.15, "failed": 0.0}.get(status, 0.0)


def score_pdf_table(
    obj: dict[str, Any],
    pdf: dict[str, Any],
    nearby_text: str,
    body_text: str,
) -> dict[str, Any]:
    chunk_page = obj.get("page")
    pdf_page = pdf.get("page_number")
    page_distance: int | str = ""
    page_score = 0.0
    if isinstance(chunk_page, int) and isinstance(pdf_page, int):
        page_distance = abs(chunk_page - pdf_page)
        if page_distance == 0:
            page_score = 1.0
        elif page_distance == 1:
            page_score = 0.45

    caption = obj.get("caption") or ""
    pdf_text = " ".join([pdf.get("table_text", ""), pdf.get("text_preview", "")])
    caption_overlap = caption_text_overlap_score(caption, pdf_text)
    nearby_overlap = caption_text_overlap_score(nearby_text, pdf_text)
    body_overlap = caption_text_overlap_score(body_text, pdf_text)
    chunk_key = normalize_table_id(obj.get("table_id") or "")
    pdf_key = normalize_table_id(table_id_from_text(pdf_text))
    table_id_match = bool(chunk_key and pdf_key and chunk_key == pdf_key)
    strong_table_id_evidence = bool(table_id_match and table_id_near_candidate_start(obj.get("table_id") or "", pdf_text))
    table_id_conflict = bool(chunk_key and pdf_key and chunk_key != pdf_key)
    l_status = layout_status(pdf)
    l_score = layout_score(pdf)
    same_page = page_distance == 0
    score = (
        0.26 * page_score
        + 0.24 * (1.0 if table_id_match else 0.0)
        + 0.2 * caption_overlap
        + 0.12 * nearby_overlap
        + 0.08 * body_overlap
        + 0.1 * l_score
    )
    if table_id_conflict:
        score -= 0.25
    if l_status == "likely_false_positive":
        score -= 0.15
    return {
        "score": round(max(0.0, min(score, 1.0)), 4),
        "caption_overlap": caption_overlap,
        "nearby_overlap": nearby_overlap,
        "body_overlap": body_overlap,
        "text_overlap_score": round(max(caption_overlap, nearby_overlap, body_overlap), 4),
        "page_distance": page_distance,
        "same_page": same_page,
        "table_id_match": table_id_match,
        "strong_table_id_evidence": strong_table_id_evidence,
        "table_id_conflict": table_id_conflict,
        "pdf_table_key": pdf_key,
        "layout_quality_status": l_status,
    }


def basis_and_blockers(
    obj: dict[str, Any],
    pdf: dict[str, Any],
    score: dict[str, Any],
    same_page_count: int,
) -> tuple[list[str], list[str]]:
    basis = ["same_doc"]
    blockers: list[str] = []
    if score["same_page"]:
        basis.append("same_page")
    if score["strong_table_id_evidence"]:
        basis.append("same_table_id")
    if score["caption_overlap"] >= 0.12:
        basis.append("caption_overlap")
    if score["nearby_overlap"] >= 0.12:
        basis.append("nearby_text_overlap")
    if score["body_overlap"] >= 0.08:
        basis.append("body_text_overlap")
    if score["same_page"] and pdf.get("table_order_on_page") is not None:
        basis.append("table_order_on_page")
    if score["layout_quality_status"] == "usable":
        basis.append("layout_quality_usable")

    if same_page_count > 1:
        blockers.append("multiple_pdf_tables_same_page")
    if score["layout_quality_status"] in {"weak", "likely_false_positive", "failed"}:
        blockers.append("low_layout_quality")
    if score["table_id_conflict"]:
        blockers.append("table_id_conflict")
    if score["table_id_match"] and not score["strong_table_id_evidence"]:
        blockers.append("table_id_only_body_reference_risk")
    if score["text_overlap_score"] < 0.12 and not score["table_id_match"]:
        blockers.append("weak_text_overlap")
    return basis, blockers


def empty_alignment_row(
    obj: dict[str, Any],
    status: str,
    confidence: str,
    blockers: list[str],
    notes: str,
) -> dict[str, str]:
    return {
        "chunk_table_object_id": obj.get("table_object_id", ""),
        "doc_id": obj.get("doc_id", ""),
        "table_id": obj.get("table_id", ""),
        "chunk_page": str(obj.get("page", "")),
        "chunk_validation_status": obj.get("validation_status", ""),
        "pdfplumber_table_id": "",
        "pdf_page": "",
        "pdf_strategy": "",
        "pdf_table_bbox": "",
        "layout_quality_status": "not_evaluable",
        "alignment_status": status,
        "alignment_confidence": confidence,
        "alignment_score": "0.0",
        "alignment_basis": "same_doc" if status == "no_pdf_table_found" else "none",
        "alignment_blockers": ";".join(blockers) if blockers else "none",
        "needs_manual_alignment_review": bool_text(status != "no_pdf_table_found"),
        "notes": notes,
    }


def alignment_from_candidate(
    obj: dict[str, Any],
    pdf: dict[str, Any],
    score: dict[str, Any],
    status: str,
    confidence: str,
    basis: list[str],
    blockers: list[str],
    manual_review: bool,
    notes: str,
) -> dict[str, str]:
    return {
        "chunk_table_object_id": obj.get("table_object_id", ""),
        "doc_id": obj.get("doc_id", ""),
        "table_id": obj.get("table_id", ""),
        "chunk_page": str(obj.get("page", "")),
        "chunk_validation_status": obj.get("validation_status", ""),
        "pdfplumber_table_id": pdf.get("pdfplumber_table_id", ""),
        "pdf_page": str(pdf.get("page_number", "")),
        "pdf_strategy": pdf.get("strategy", ""),
        "pdf_table_bbox": json.dumps(pdf.get("bbox"), ensure_ascii=False),
        "layout_quality_status": score["layout_quality_status"],
        "alignment_status": status,
        "alignment_confidence": confidence,
        "alignment_score": str(score["score"]),
        "alignment_basis": ";".join(basis) if basis else "none",
        "alignment_blockers": ";".join(sorted(set(blockers))) if blockers else "none",
        "needs_manual_alignment_review": bool_text(manual_review),
        "notes": notes,
    }


def choose_alignment(
    obj: dict[str, Any],
    pdf_tables: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]] | None = None,
    pdf_missing: bool = False,
) -> dict[str, str]:
    chunks_by_id = chunks_by_id or {}
    if pdf_missing:
        return empty_alignment_row(
            obj,
            "no_pdf_table_found",
            "none",
            ["no_pdf_table_found"],
            "未找到该 doc_id 的 PDF，不能伪造 layout。",
        )
    doc_pdfs = [pdf for pdf in pdf_tables if pdf.get("doc_id") == obj.get("doc_id")]
    if not doc_pdfs:
        return empty_alignment_row(
            obj,
            "no_pdf_table_found",
            "none",
            ["no_pdf_table_found"],
            "raw pdfplumber extraction 中没有该 doc_id 的表格。",
        )

    nearby_text = chunk_nearby_text(obj, chunks_by_id)
    body_text = chunk_body_text(obj, chunks_by_id)
    scored = [(pdf, score_pdf_table(obj, pdf, nearby_text, body_text)) for pdf in doc_pdfs]
    scored.sort(key=lambda item: item[1]["score"], reverse=True)
    top_pdf, top_score = scored[0]
    same_page = [(pdf, score) for pdf, score in scored if score["same_page"]]
    same_page_count = len(same_page)
    basis, blockers = basis_and_blockers(obj, top_pdf, top_score, same_page_count)
    layout_usable = top_score["layout_quality_status"] == "usable"

    if top_score["table_id_conflict"]:
        blockers.append("table_id_conflict")
        return alignment_from_candidate(
            obj,
            top_pdf,
            top_score,
            "conflict",
            "low",
            basis,
            blockers,
            True,
            "chunk table_id 与 pdfplumber table_id 信号冲突，需要人工 review。",
        )

    cannot_disambiguate_same_page = (
        same_page_count > 1
        and not top_score["table_id_match"]
        and top_score["text_overlap_score"] < 0.14
    )
    if cannot_disambiguate_same_page:
        blockers.extend(["multiple_pdf_tables_same_page", "weak_text_overlap"])
        return alignment_from_candidate(
            obj,
            top_pdf,
            top_score,
            "multiple_pdf_tables",
            "low",
            basis,
            blockers,
            True,
            "同页存在多个 pdfplumber 候选且 caption/table_id/body 信号弱，需要人工对齐。",
        )

    if top_score["same_page"] and top_score["strong_table_id_evidence"] and layout_usable:
        return alignment_from_candidate(
            obj,
            top_pdf,
            top_score,
            "matched",
            "high",
            basis,
            blockers,
            False,
            "high 需要 same_doc + same_page + caption/table title 附近的 same_table_id 且 layout_quality usable；仍不等于 extraction correct。",
        )

    has_text_basis = top_score["text_overlap_score"] >= 0.12
    if top_score["same_page"] and has_text_basis and layout_usable:
        return alignment_from_candidate(
            obj,
            top_pdf,
            top_score,
            "matched",
            "medium",
            basis,
            blockers,
            False,
            "medium 需要 same_doc + same_page + caption/nearby/body overlap 且 layout_quality usable。",
        )

    if top_score["text_overlap_score"] >= 0.28 and not top_score["same_page"]:
        blockers.append("page_only_match")
        return alignment_from_candidate(
            obj,
            top_pdf,
            top_score,
            "caption_only_match",
            "low",
            basis,
            blockers,
            True,
            "caption/text overlap 存在但页码不稳，不能直接用于 confirmed。",
        )

    if top_score["same_page"]:
        blockers.append("page_only_match")
        return alignment_from_candidate(
            obj,
            top_pdf,
            top_score,
            "page_only_match",
            "low",
            basis,
            blockers,
            True,
            "page_only_match 只是候选页对齐，Phase7C-2 不再默认可信。",
        )

    return alignment_from_candidate(
        obj,
        top_pdf,
        top_score,
        "conflict",
        "low",
        basis,
        blockers or ["weak_text_overlap"],
        True,
        "最佳候选与 page/caption/body 信号不一致，需要人工 review。",
    )


def write_alignment_report(
    rows: list[dict[str, str]],
    report_path: Path,
    phase7c_rows: list[dict[str, str]] | None = None,
) -> None:
    phase7c_rows = phase7c_rows or []
    status_counts = Counter(row["alignment_status"] for row in rows)
    confidence_counts = Counter(row["alignment_confidence"] for row in rows)
    blocker_counts: Counter[str] = Counter()
    for row in rows:
        blockers = [] if row["alignment_blockers"] == "none" else row["alignment_blockers"].split(";")
        blocker_counts.update(blockers)
    manual_count = sum(1 for row in rows if row["needs_manual_alignment_review"] == "true")
    page_only_count = status_counts.get("page_only_match", 0)
    phase7c_status = Counter(row.get("alignment_status", "") for row in phase7c_rows)
    phase7c_confidence = Counter(row.get("alignment_confidence", "") for row in phase7c_rows)

    lines = [
        "# chunk 与 pdfplumber alignment gate 报告",
        "",
        "## 1. 对齐目标",
        "",
        "本报告将 Phase7B-2 chunk table_object 与 Phase7C-2 pdfplumber raw table 对齐，并显式输出 confidence、score、basis、blockers 与 manual review gate。",
        "",
        "Phase7C-2 的核心收紧是：`page_only_match` 固定为 low confidence/manual review；`high` 必须同时满足 same_doc、same_page、same_table_id 和 usable layout；likely_false_positive layout 不能 high。",
        "",
        "## 2. alignment_status 统计",
        "",
        "| alignment_status | 数量 |",
        "|---|---:|",
    ]
    for status in [
        "matched",
        "page_only_match",
        "caption_only_match",
        "multiple_pdf_tables",
        "conflict",
        "no_pdf_table_found",
        "not_evaluable",
    ]:
        lines.append(f"| `{status}` | {status_counts.get(status, 0)} |")
    lines.extend(["", "## 3. alignment_confidence 统计", "", "| confidence | 数量 |", "|---|---:|"])
    for confidence in ["high", "medium", "low", "none"]:
        lines.append(f"| `{confidence}` | {confidence_counts.get(confidence, 0)} |")
    lines.extend(
        [
            "",
            "## 4. manual review 与 blocker 统计",
            "",
            f"- needs_manual_alignment_review 数量：{manual_count}",
            f"- page_only_match 数量：{page_only_count}",
            f"- low_layout_quality blocker 数量：{blocker_counts.get('low_layout_quality', 0)}",
            f"- multiple_pdf_tables_same_page 数量：{blocker_counts.get('multiple_pdf_tables_same_page', 0)}",
            "",
            "| blocker | 数量 |",
            "|---|---:|",
        ]
    )
    for blocker, count in blocker_counts.most_common():
        lines.append(f"| `{blocker}` | {count} |")
    if not blocker_counts:
        lines.append("| none | 0 |")
    lines.extend(
        [
            "",
            "## 5. 与 Phase7C alignment 的主要变化",
            "",
            f"- Phase7C alignment_status：{dict(phase7c_status)}",
            f"- Phase7C alignment_confidence：{dict(phase7c_confidence)}",
            f"- Phase7C-2 alignment_status：{dict(status_counts)}",
            f"- Phase7C-2 alignment_confidence：{dict(confidence_counts)}",
            "- page_only_match 不再给 medium，也不允许直接进入 pass-ready hybrid object。",
            "- layout_quality_status 进入 gate 后，低质量 layout 可作为 blocker 暴露，而不是隐藏在主 table_object 中。",
            "",
            "## 6. 需要人工对齐复核的对象",
            "",
            "| table_object_id | doc_id | table_id | status | confidence | blockers | notes |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    manual_rows = [row for row in rows if row["needs_manual_alignment_review"] == "true"]
    for row in manual_rows:
        lines.append(
            f"| `{row['chunk_table_object_id']}` | `{row['doc_id']}` | `{row['table_id']}` | `{row['alignment_status']}` | `{row['alignment_confidence']}` | {row['alignment_blockers']} | {row['notes']} |"
        )
    if not manual_rows:
        lines.append("| none |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## 7. 口径说明",
            "",
            "- 本脚本只读取 Phase7B-2 table_objects、pdfplumber raw tables 和 official chunks；未读取 BM25 index，未访问 Milvus。",
            "- alignment_score 仅用于排序和审阅，不写入主 hybrid table_object。",
            "- matched 只表示候选对齐，不等于 pdfplumber extraction correct。",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    objects = load_jsonl(args.table_objects)
    pdf_tables = load_jsonl(args.pdfplumber_raw)
    chunks_by_id = load_chunks_by_id(args.chunks, SMOKE_DOC_IDS)
    rows: list[dict[str, str]] = []
    for obj in objects:
        doc_id = obj.get("doc_id")
        if doc_id not in SMOKE_DOC_IDS:
            continue
        rows.append(
            choose_alignment(
                obj,
                pdf_tables,
                chunks_by_id,
                pdf_missing=find_pdf(doc_id, args.pdf_dir) is None,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    phase7c_rows = load_csv(args.phase7c_alignment)
    write_alignment_report(rows, args.report, phase7c_rows)
    print(
        json.dumps(
            {
                "alignment_rows": len(rows),
                "status_counts": dict(Counter(row["alignment_status"] for row in rows)),
                "confidence_counts": dict(Counter(row["alignment_confidence"] for row in rows)),
                "output": rel(args.output),
                "report": rel(args.report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align Phase7B-2 chunk tables to pdfplumber tables.")
    parser.add_argument("--table-objects", type=Path, default=DEFAULT_TABLE_OBJECTS_PATH)
    parser.add_argument("--pdfplumber-raw", type=Path, default=DEFAULT_PDFPLUMBER_RAW_PATH)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--phase7c-alignment", type=Path, default=DEFAULT_PHASE7C_ALIGNMENT_PATH)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    args = parser.parse_args()
    args.table_objects = resolve_path(args.table_objects)
    args.pdfplumber_raw = resolve_path(args.pdfplumber_raw)
    args.chunks = resolve_path(args.chunks)
    args.output = resolve_path(args.output)
    args.report = resolve_path(args.report)
    args.phase7c_alignment = resolve_path(args.phase7c_alignment)
    args.pdf_dir = resolve_path(args.pdf_dir)
    return args


if __name__ == "__main__":
    run(parse_args())
