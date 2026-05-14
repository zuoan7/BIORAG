#!/usr/bin/env python3
from __future__ import annotations

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag.domain.config import Settings
from src.synbio_rag.ingestion.kb_builder import KnowledgeBaseBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build round-1 knowledge base.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--table-enhancement",
        action="store_true",
        help="Enable conservative table-like paragraph enhancement for this build.",
    )
    group.add_argument(
        "--no-table-enhancement",
        action="store_true",
        help="Disable table enhancement for this build.",
    )
    parser.add_argument("--table-enhancement-output-suffix")
    parser.add_argument("--table-enhancement-audit-dir")
    parser.add_argument("--table-enhancement-dry-run", action="store_true")
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.table_enhancement:
        settings.table_enhancement.enabled = True
    if args.no_table_enhancement:
        settings.table_enhancement.enabled = False
    if args.table_enhancement_output_suffix:
        settings.table_enhancement.output_suffix = args.table_enhancement_output_suffix
    if args.table_enhancement_audit_dir:
        settings.table_enhancement.audit_root = str(Path(args.table_enhancement_audit_dir).resolve())
        setattr(settings.table_enhancement, "audit_dir_exact", True)
    if args.table_enhancement_dry_run:
        settings.table_enhancement.enabled = True
        settings.table_enhancement.dry_run = True

    builder = KnowledgeBaseBuilder(settings)
    builder.build_round1()


if __name__ == "__main__":
    main()
