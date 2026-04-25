from __future__ import annotations

import subprocess
from pathlib import Path

from ..domain.config import Settings


class KnowledgeBaseBuilder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = Path(__file__).resolve().parents[3]

    def build_round1(self) -> None:
        self.settings.ensure_directories()
        self._run(
            [
                "python",
                "scripts/ingestion/pdf_to_structured.py",
                "--input_dir",
                self.settings.kb.paper_dir,
                "--output_dir",
                self.settings.kb.parsed_dir,
                "--also_txt",
            ]
        )
        self._run(
            [
                "python",
                "scripts/ingestion/preprocess_and_chunk.py",
                "--input_dir",
                self.settings.kb.parsed_dir,
                "--output_dir",
                self.settings.kb.chunk_dir,
                "--chunk_size",
                str(self.settings.kb.chunk_size),
                "--chunk_overlap",
                str(self.settings.kb.chunk_overlap),
            ]
        )
        self._run(
            [
                "python",
                "scripts/ingestion/import_to_milvus.py",
                "--jsonl",
                self.settings.kb.chunk_jsonl,
                "--collection_name",
                self.settings.retrieval.collection_name,
                "--milvus_uri",
                self.settings.retrieval.milvus_uri,
                "--embedding",
                "bge-m3",
                "--model_path",
                self.settings.kb.embedding_model_path,
                "--dim",
                str(self.settings.kb.embedding_dim),
            ]
        )

    def _run(self, cmd: list[str]) -> None:
        subprocess.run(cmd, cwd=self.root, check=True)
