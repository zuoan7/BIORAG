#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit parsed_raw_v4 JSON outputs.

Usage:
    python scripts/diagnostics/audit_parsed_raw_v4.py tmp/v4_phase1_parsed_raw
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def audit_file(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    diagnostics = data.get("diagnostics", {})
    pages = data.get("pages", [])
    order_strategy_distribution = Counter(
        item.get("selected_order_strategy", "unknown")
        for item in diagnostics.get("page_layout_diagnostics", [])
    )

    return {
        "doc_id": data.get("doc_id", path.stem),
        "total_pages": data.get("total_pages", len(pages)),
        "two_column_pages": diagnostics.get("two_column_pages", []),
        "image_block_count": diagnostics.get("image_block_count", 0),
        "figure_caption_candidate_count": diagnostics.get("figure_caption_candidate_count", 0),
        "table_caption_candidate_count": diagnostics.get("table_caption_candidate_count", 0),
        "journal_preproof_noise_count": diagnostics.get("stripped_noise_line_count", 0),
        "pages_with_empty_text": [
            page.get("page")
            for page in pages
            if not (page.get("text") or "").strip()
        ],
        "order_strategy_distribution": dict(order_strategy_distribution),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit parsed_raw_v4 diagnostics")
    parser.add_argument("parsed_raw_dir", help="Directory containing parsed_raw_v4 JSON files")
    args = parser.parse_args()

    input_dir = Path(args.parsed_raw_dir)
    json_files = sorted(path for path in input_dir.glob("*.json") if path.is_file())
    if not json_files:
        print(f"[WARN] no JSON files found: {input_dir}")
        return

    for path in json_files:
        row = audit_file(path)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
