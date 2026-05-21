from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..domain.config import Settings
from .table_enhancement import (
    config_from_settings,
    derive_suffixed_path,
    make_run_id,
    run_table_enhancement,
)


@dataclass(frozen=True)
class Round1BuildPaths:
    parsed_input_dir: Path
    chunk_dir: Path
    chunk_jsonl: Path
    parent_chunk_jsonl: Path
    child_chunk_jsonl: Path
    parent_index_jsonl: Path
    bm25_cache_path: Path
    milvus_uri: str
    collection_name: str
    table_enhancement_output_dir: Path | None = None
    table_enhancement_audit_dir: Path | None = None


class KnowledgeBaseBuilder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = Path(__file__).resolve().parents[3]

    def build_round1(self) -> None:
        self.settings.ensure_directories()
        table_config = self.settings.table_enhancement
        build_paths = self._round1_build_paths()
        self._run(
            [
                "python",
                "scripts/ingestion/pdf_to_structured.py",
                "--input_dir",
                self.settings.kb.paper_dir,
                "--output_dir",
                self.settings.kb.parsed_raw_dir,
                "--also_txt",
            ]
        )
        self._run(
            [
                "python",
                "scripts/ingestion/clean_parsed_structure.py",
                "--input_dir",
                self.settings.kb.parsed_raw_dir,
                "--output_dir",
                self.settings.kb.parsed_dir,
                "--preview_dir",
                self.settings.kb.parsed_preview_dir,
            ]
        )
        print(
            "[table_enhancement] "
            f"enabled={table_config.enabled} dry_run={table_config.dry_run}"
        )
        if table_config.enabled:
            if build_paths.table_enhancement_output_dir is None or build_paths.table_enhancement_audit_dir is None:
                raise RuntimeError("table enhancement paths were not initialized")
            result = run_table_enhancement(
                input_dir=self.settings.kb.parsed_dir,
                output_dir=build_paths.table_enhancement_output_dir,
                audit_dir=build_paths.table_enhancement_audit_dir,
                config=config_from_settings(table_config),
                process_all_docs=True,
            )
            print(
                "[table_enhancement] "
                f"input_dir={self.settings.kb.parsed_dir} "
                f"output_dir={build_paths.table_enhancement_output_dir} "
                f"audit_dir={build_paths.table_enhancement_audit_dir} "
                f"associations={result.association_count} "
                f"safety_gate_passed={result.safety_gate_passed}"
            )
            if not result.safety_gate_passed:
                raise RuntimeError(
                    "table enhancement safety gate failed; see "
                    f"{build_paths.table_enhancement_audit_dir}"
                )
        else:
            print(
                "[table_enhancement] "
                f"input_dir={self.settings.kb.parsed_dir} "
                "output_dir=n/a audit_dir=n/a associations=0 safety_gate_passed=n/a"
            )
        self._run(
            [
                "python",
                "scripts/ingestion/preprocess_and_chunk.py",
                "--input_dir",
                str(build_paths.parsed_input_dir),
                "--output_dir",
                str(build_paths.chunk_dir),
                "--chunk_size",
                str(self.settings.kb.chunk_size),
                "--chunk_overlap",
                str(self.settings.kb.chunk_overlap),
            ]
        )
        self._run(
            [
                "python",
                "scripts/ingestion/build_parent_child_index.py",
                "--chunks",
                str(build_paths.chunk_jsonl),
                "--parent-output",
                str(build_paths.parent_chunk_jsonl),
                "--child-output",
                str(build_paths.child_chunk_jsonl),
                "--parent-index-output",
                str(build_paths.parent_index_jsonl),
                "--child-size",
                "160",
                "--child-overlap",
                "40",
            ]
        )
        self._run(
            [
                "python",
                "scripts/ingestion/import_to_milvus.py",
                "--jsonl",
                str(build_paths.child_chunk_jsonl),
                "--collection_name",
                build_paths.collection_name,
                "--milvus_uri",
                build_paths.milvus_uri,
                "--embedding",
                "bge-m3",
                "--model_path",
                self.settings.kb.embedding_model_path,
                "--dim",
                str(self.settings.kb.embedding_dim),
                "--rebuild",
            ]
        )

    def _run(self, cmd: list[str]) -> None:
        subprocess.run(cmd, cwd=self.root, check=True)

    def _round1_build_paths(self) -> Round1BuildPaths:
        table_config = self.settings.table_enhancement
        parsed_dir = Path(self.settings.kb.parsed_dir)
        chunk_dir = Path(self.settings.kb.chunk_dir)
        chunk_jsonl = Path(self.settings.kb.chunk_jsonl)
        parent_chunk_jsonl = chunk_jsonl.with_name("parent_chunks.jsonl")
        child_chunk_jsonl = chunk_jsonl.with_name("child_chunks.jsonl")
        parent_index_jsonl = chunk_jsonl.with_name("parent_index.jsonl")
        bm25_cache_path = Path(self.settings.retrieval.bm25_cache_path)
        if not table_config.enabled:
            return Round1BuildPaths(
                parsed_input_dir=parsed_dir,
                chunk_dir=chunk_dir,
                chunk_jsonl=chunk_jsonl,
                parent_chunk_jsonl=parent_chunk_jsonl,
                child_chunk_jsonl=child_chunk_jsonl,
                parent_index_jsonl=parent_index_jsonl,
                bm25_cache_path=bm25_cache_path,
                milvus_uri=self.settings.retrieval.milvus_uri,
                collection_name=self.settings.retrieval.collection_name,
            )

        suffix = table_config.output_suffix
        enhanced_parsed_dir = derive_suffixed_path(parsed_dir, suffix)
        enhanced_chunk_dir = derive_suffixed_path(chunk_dir, suffix)
        enhanced_chunk_jsonl = enhanced_chunk_dir / chunk_jsonl.name
        enhanced_parent_chunk_jsonl = enhanced_chunk_dir / parent_chunk_jsonl.name
        enhanced_child_chunk_jsonl = enhanced_chunk_dir / child_chunk_jsonl.name
        enhanced_parent_index_jsonl = enhanced_chunk_dir / parent_index_jsonl.name
        enhanced_bm25_cache_path = enhanced_chunk_dir / bm25_cache_path.name
        enhanced_milvus_uri = self._suffixed_milvus_uri(self.settings.retrieval.milvus_uri, suffix)
        enhanced_collection = f"{self.settings.retrieval.collection_name}_{suffix}"
        audit_dir = Path(table_config.audit_root)
        if not getattr(table_config, "audit_dir_exact", False):
            audit_dir = audit_dir / make_run_id()

        self._guard_isolated_paths(
            parsed_dir=parsed_dir,
            chunk_dir=chunk_dir,
            chunk_jsonl=chunk_jsonl,
            bm25_cache_path=bm25_cache_path,
            enhanced_parsed_dir=enhanced_parsed_dir,
            enhanced_chunk_dir=enhanced_chunk_dir,
            enhanced_chunk_jsonl=enhanced_chunk_jsonl,
            enhanced_parent_chunk_jsonl=enhanced_parent_chunk_jsonl,
            enhanced_child_chunk_jsonl=enhanced_child_chunk_jsonl,
            enhanced_parent_index_jsonl=enhanced_parent_index_jsonl,
            enhanced_bm25_cache_path=enhanced_bm25_cache_path,
            enhanced_milvus_uri=enhanced_milvus_uri,
            enhanced_collection=enhanced_collection,
        )

        parsed_input = parsed_dir if table_config.dry_run else enhanced_parsed_dir
        return Round1BuildPaths(
            parsed_input_dir=parsed_input,
            chunk_dir=chunk_dir if table_config.dry_run else enhanced_chunk_dir,
            chunk_jsonl=chunk_jsonl if table_config.dry_run else enhanced_chunk_jsonl,
            parent_chunk_jsonl=parent_chunk_jsonl if table_config.dry_run else enhanced_parent_chunk_jsonl,
            child_chunk_jsonl=child_chunk_jsonl if table_config.dry_run else enhanced_child_chunk_jsonl,
            parent_index_jsonl=parent_index_jsonl if table_config.dry_run else enhanced_parent_index_jsonl,
            bm25_cache_path=bm25_cache_path if table_config.dry_run else enhanced_bm25_cache_path,
            milvus_uri=self.settings.retrieval.milvus_uri if table_config.dry_run else enhanced_milvus_uri,
            collection_name=self.settings.retrieval.collection_name if table_config.dry_run else enhanced_collection,
            table_enhancement_output_dir=enhanced_parsed_dir,
            table_enhancement_audit_dir=audit_dir,
        )

    @staticmethod
    def _suffixed_milvus_uri(uri: str, suffix: str) -> str:
        if uri.startswith(("http://", "https://", "unix://", "tcp://")):
            return uri
        return str(derive_suffixed_path(uri, suffix))

    @staticmethod
    def _guard_isolated_paths(
        *,
        parsed_dir: Path,
        chunk_dir: Path,
        chunk_jsonl: Path,
        bm25_cache_path: Path,
        enhanced_parsed_dir: Path,
        enhanced_chunk_dir: Path,
        enhanced_chunk_jsonl: Path,
        enhanced_parent_chunk_jsonl: Path,
        enhanced_child_chunk_jsonl: Path,
        enhanced_parent_index_jsonl: Path,
        enhanced_bm25_cache_path: Path,
        enhanced_milvus_uri: str,
        enhanced_collection: str,
    ) -> None:
        path_pairs = [
            ("parsed_dir", parsed_dir, enhanced_parsed_dir),
            ("chunk_dir", chunk_dir, enhanced_chunk_dir),
            ("chunk_jsonl", chunk_jsonl, enhanced_chunk_jsonl),
            ("parent_chunk_jsonl", Path(chunk_jsonl).with_name("parent_chunks.jsonl"), enhanced_parent_chunk_jsonl),
            ("child_chunk_jsonl", Path(chunk_jsonl).with_name("child_chunks.jsonl"), enhanced_child_chunk_jsonl),
            ("parent_index_jsonl", Path(chunk_jsonl).with_name("parent_index.jsonl"), enhanced_parent_index_jsonl),
            ("bm25_cache_path", bm25_cache_path, enhanced_bm25_cache_path),
        ]
        for name, baseline, enhanced in path_pairs:
            if baseline.resolve() == enhanced.resolve():
                raise ValueError(f"table enhancement {name} target collides with baseline path")
        if not enhanced_collection or enhanced_collection == "synbio_papers":
            raise ValueError("table enhancement collection_name must be isolated")
        if enhanced_milvus_uri and enhanced_milvus_uri == "":
            raise ValueError("table enhancement milvus_uri must not be empty")
