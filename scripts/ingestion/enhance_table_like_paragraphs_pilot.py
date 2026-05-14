#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for conservative table-like paragraph enhancement."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag.ingestion.table_enhancement import (  # noqa: E402
    TableEnhancementRunConfig,
    read_selected_docs,
    run_table_enhancement,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conservatively mark table-like paragraphs near table captions."
    )
    parser.add_argument("--selected_docs")
    parser.add_argument(
        "--all_docs",
        action="store_true",
        help="Process every *.json document in input_dir.",
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--audit_dir", required=True)
    parser.add_argument("--window_after_caption", type=int, default=5)
    parser.add_argument("--window_before_caption", type=int, default=1)
    parser.add_argument("--max_associated_blocks_per_caption", type=int, default=5)
    parser.add_argument("--min_confidence", default="low", choices=("low", "medium", "high"))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    selected_doc_ids = None
    process_all_docs = bool(args.all_docs)
    if args.selected_docs:
        selected_doc_ids = read_selected_docs(Path(args.selected_docs))
        process_all_docs = False
    elif not args.all_docs:
        parser.error("--selected_docs is required unless --all_docs is set")

    result = run_table_enhancement(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        audit_dir=Path(args.audit_dir),
        config=TableEnhancementRunConfig(
            window_after_caption=args.window_after_caption,
            window_before_caption=args.window_before_caption,
            max_associated_blocks_per_caption=args.max_associated_blocks_per_caption,
            min_confidence=args.min_confidence,
            dry_run=args.dry_run,
        ),
        selected_doc_ids=selected_doc_ids,
        process_all_docs=process_all_docs,
    )
    _write_pilot_aliases(Path(args.audit_dir))
    print(json.dumps(result.to_summary_dict(), ensure_ascii=False, indent=2))


def _write_pilot_aliases(audit_dir: Path) -> None:
    aliases = {
        "association_audit.csv": "pilot_association_audit.csv",
        "false_positive_review.md": "pilot_association_examples.md",
        "association_summary.json": "pilot_enhancement_summary.json",
    }
    for src_name, dst_name in aliases.items():
        src = audit_dir / src_name
        dst = audit_dir / dst_name
        if src.exists():
            shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
