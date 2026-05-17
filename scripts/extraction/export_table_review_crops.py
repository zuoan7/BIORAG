#!/usr/bin/env python3
"""Export Phase7G PDF crop PNGs with pdfplumber only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extraction import extract_tables_pdfplumber_v1 as pdf_extract


DEFAULT_REVIEW_POOL = ROOT / "data/experiments/v7_phase7_expanded_table_review_pack/candidate_pool_raw.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data/experiments/v7_phase7_expanded_table_review_pack/pdf_crops"
DEFAULT_PDF_DIR = ROOT / "data/paper_round1/paper"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_bbox(bbox: Any, page_width: float, page_height: float) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(value) for value in bbox]
    except Exception:
        return None
    pad = 12.0
    x0 = max(0.0, x0 - pad)
    y0 = max(0.0, y0 - pad)
    x1 = min(page_width, x1 + pad)
    y1 = min(page_height, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def export_crop(candidate: dict[str, Any], output_dir: Path, pdf_dir: Path, resolution: int = 150) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_id = candidate.get("candidate_id", "")
    if not candidate_id:
        return {"candidate_id": "", "pdf_crop_path": "", "crop_status": "failed", "crop_error": "missing_candidate_id"}
    pdfplumber, _, import_error = pdf_extract.import_pdfplumber()
    if not pdfplumber:
        return {
            "candidate_id": candidate_id,
            "pdf_crop_path": "",
            "crop_status": "failed",
            "crop_error": f"pdfplumber_unavailable:{import_error}",
        }
    doc_id = candidate.get("doc_id", "")
    pdf_path = resolve_path(Path(candidate.get("pdf_path") or "")) if candidate.get("pdf_path") else None
    if not pdf_path or not pdf_path.exists():
        pdf_path = pdf_extract.find_pdf(doc_id, pdf_dir)
    if not pdf_path:
        return {"candidate_id": candidate_id, "pdf_crop_path": "", "crop_status": "failed", "crop_error": "pdf_missing"}
    page_number = int(candidate.get("page") or 0)
    if page_number <= 0:
        return {"candidate_id": candidate_id, "pdf_crop_path": "", "crop_status": "failed", "crop_error": "page_missing"}
    output_path = output_dir / f"{candidate_id}.png"
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number > len(pdf.pages):
                return {
                    "candidate_id": candidate_id,
                    "pdf_crop_path": "",
                    "crop_status": "failed",
                    "crop_error": "page_out_of_range",
                }
            page = pdf.pages[page_number - 1]
            bbox = normalize_bbox(candidate.get("crop_bbox") or candidate.get("pdf_table_bbox"), page.width, page.height)
            crop_page = page.crop(bbox) if bbox else page
            image = crop_page.to_image(resolution=resolution)
            image.save(output_path)
    except Exception as exc:
        return {
            "candidate_id": candidate_id,
            "pdf_crop_path": rel(output_path),
            "crop_status": "failed",
            "crop_error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "candidate_id": candidate_id,
        "pdf_crop_path": rel(output_path),
        "crop_status": "ok",
        "crop_error": "",
    }


def export_crops(candidates: list[dict[str, Any]], output_dir: Path, pdf_dir: Path, resolution: int = 150) -> dict[str, dict[str, str]]:
    results: dict[str, dict[str, str]] = {}
    for candidate in candidates:
        result = export_crop(candidate, output_dir, pdf_dir, resolution)
        results[result["candidate_id"]] = result
    return results


def run(args: argparse.Namespace) -> None:
    candidates = load_jsonl(args.review_pool)
    candidates = [row for row in candidates if row.get("review_priority") != "auto_excluded"]
    results = export_crops(candidates, args.output_dir, args.pdf_dir, args.resolution)
    print(
        json.dumps(
            {
                "crops_ok": sum(1 for row in results.values() if row.get("crop_status") == "ok"),
                "crops_failed": sum(1 for row in results.values() if row.get("crop_status") == "failed"),
                "output_dir": rel(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Phase7G table review PDF crop PNGs.")
    parser.add_argument("--review-pool", type=Path, default=DEFAULT_REVIEW_POOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--resolution", type=int, default=150)
    args = parser.parse_args()
    args.review_pool = resolve_path(args.review_pool)
    args.output_dir = resolve_path(args.output_dir)
    args.pdf_dir = resolve_path(args.pdf_dir)
    return args


if __name__ == "__main__":
    run(parse_args())
