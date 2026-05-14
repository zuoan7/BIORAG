from __future__ import annotations

import json
from pathlib import Path

from src.synbio_rag.domain.config import Settings
from src.synbio_rag.ingestion.kb_builder import KnowledgeBaseBuilder
from src.synbio_rag.ingestion.table_enhancement import (
    TableEnhancementRunConfig,
    run_table_enhancement,
)


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.kb.parsed_dir = str(tmp_path / "parsed_clean")
    settings.kb.chunk_dir = str(tmp_path / "chunks")
    settings.kb.chunk_jsonl = str(tmp_path / "chunks" / "chunks.jsonl")
    settings.retrieval.bm25_cache_path = str(tmp_path / "chunks" / "bm25_index.json")
    settings.retrieval.milvus_uri = str(tmp_path / "milvus" / "papers.db")
    settings.retrieval.collection_name = "synbio_papers"
    settings.table_enhancement.audit_root = str(tmp_path / "reports" / "table_enhancement")
    return settings


def test_default_off_uses_old_parsed_clean_path(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    paths = KnowledgeBaseBuilder(settings)._round1_build_paths()

    assert paths.parsed_input_dir == Path(settings.kb.parsed_dir)
    assert paths.chunk_dir == Path(settings.kb.chunk_dir)
    assert paths.chunk_jsonl == Path(settings.kb.chunk_jsonl)
    assert paths.table_enhancement_output_dir is None


def test_enabled_uses_isolated_enhanced_paths(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.table_enhancement.enabled = True
    paths = KnowledgeBaseBuilder(settings)._round1_build_paths()

    assert paths.parsed_input_dir.name == "parsed_clean_table_enhanced"
    assert paths.chunk_dir.name == "chunks_table_enhanced"
    assert paths.chunk_jsonl.name == "chunks.jsonl"
    assert paths.chunk_jsonl.parent == paths.chunk_dir
    assert paths.collection_name == "synbio_papers_table_enhanced"
    assert paths.milvus_uri.endswith("papers_table_enhanced.db")
    assert paths.parsed_input_dir != Path(settings.kb.parsed_dir)


def test_dry_run_does_not_change_downstream_input_path(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.table_enhancement.enabled = True
    settings.table_enhancement.dry_run = True
    paths = KnowledgeBaseBuilder(settings)._round1_build_paths()

    assert paths.parsed_input_dir == Path(settings.kb.parsed_dir)
    assert paths.chunk_dir == Path(settings.kb.chunk_dir)
    assert paths.collection_name == settings.retrieval.collection_name
    assert paths.table_enhancement_output_dir is not None


def test_enhancement_does_not_overwrite_original_parsed_clean(tmp_path: Path) -> None:
    input_dir = tmp_path / "parsed_clean"
    output_dir = tmp_path / "parsed_clean_table_enhanced"
    audit_dir = tmp_path / "audit"
    input_dir.mkdir()
    original = {
        "doc_id": "doc1",
        "source_file": "doc1.pdf",
        "parser_stage": "parsed_clean_v4",
        "pages": [{
            "page": 1,
            "text": "Table 1. Strains.\nStrain  Plasmid  1  2  3",
            "blocks": [
                {"block_id": "b1", "type": "table_caption", "text": "Table 1. Strains.", "page": 1, "metadata": {}},
                {"block_id": "b2", "type": "paragraph", "text": "Strain  Plasmid  1  2  3", "page": 1, "metadata": {}},
            ],
        }],
    }
    src = input_dir / "doc1.json"
    src.write_text(json.dumps(original), encoding="utf-8")

    run_table_enhancement(
        input_dir=input_dir,
        output_dir=output_dir,
        audit_dir=audit_dir,
        config=TableEnhancementRunConfig(),
    )

    baseline = json.loads(src.read_text(encoding="utf-8"))
    enhanced = json.loads((output_dir / "doc1.json").read_text(encoding="utf-8"))
    assert "table_related" not in baseline["pages"][0]["blocks"][1]["metadata"]
    assert enhanced["pages"][0]["blocks"][1]["metadata"]["table_related"] is True
