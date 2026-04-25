#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag.domain.config import Settings
from src.synbio_rag.ingestion.kb_builder import KnowledgeBaseBuilder


def main() -> None:
    settings = Settings.from_env()
    builder = KnowledgeBaseBuilder(settings)
    builder.build_round1()


if __name__ == "__main__":
    main()
