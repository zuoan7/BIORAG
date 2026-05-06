"""Phase 12A 结构化索引契约升级 测试"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is on path for scripts imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ingestion.import_to_milvus import (
    METADATA_JSON_VARCHAR_MAX_LENGTH,
    RETRIEVAL_TEXT_VARCHAR_MAX_LENGTH,
    TEXT_VARCHAR_MAX_LENGTH,
    _safe_truncate,
    _truncation_stats,
    build_collection_schema,
    build_metadata_json,
    derive_content_kind,
    derive_object_type,
    infer_object_id,
)
from src.synbio_rag.domain.config import RetrievalConfig, Settings
from src.synbio_rag.domain.schemas import RetrievedChunk
from src.synbio_rag.infrastructure.vectorstores.bm25 import (
    BM25Retriever,
    _retrieval_text,
    _safe_parse_metadata_json as _bm25_safe_parse,
)
from src.synbio_rag.infrastructure.vectorstores.milvus import (
    MilvusRetriever,
    _safe_parse_metadata_json,
)


# ============================================================
# 1. Schema 字段测试
# ============================================================

class TestBuildCollectionSchema:
    def test_contains_all_phase12a_fields(self):
        schema = build_collection_schema(dim=1024)
        field_names = {f.name for f in schema.fields}

        required = {
            "retrieval_text", "content_kind", "quality_score",
            "contains_table_text", "contains_table_caption",
            "contains_figure_caption", "contains_image",
            "object_type", "object_id", "metadata_json",
        }
        missing = required - field_names
        assert not missing, f"缺少字段: {missing}"

    def test_legacy_fields_preserved(self):
        schema = build_collection_schema(dim=1024)
        field_names = {f.name for f in schema.fields}
        legacy = {"chunk_id", "doc_id", "source_file", "title", "section",
                   "page_start", "page_end", "chunk_index", "text", "embedding"}
        missing = legacy - field_names
        assert not missing, f"缺少旧字段: {missing}"

    def test_text_varchar_max_length(self):
        schema = build_collection_schema(dim=1024)
        by_name = {f.name: f for f in schema.fields}
        assert by_name["text"].params["max_length"] == TEXT_VARCHAR_MAX_LENGTH
        assert by_name["retrieval_text"].params["max_length"] == RETRIEVAL_TEXT_VARCHAR_MAX_LENGTH
        assert by_name["metadata_json"].params["max_length"] == METADATA_JSON_VARCHAR_MAX_LENGTH

    def test_text_max_length_is_16384_not_2048(self):
        """text/retrieval_text 存储上限为 16384 字符，不是 2048。"""
        schema = build_collection_schema(dim=1024)
        by_name = {f.name: f for f in schema.fields}
        assert by_name["text"].params["max_length"] == 16384
        assert by_name["retrieval_text"].params["max_length"] == 16384


# ============================================================
# 2. content_kind 推导测试
# ============================================================

class TestDeriveContentKind:
    def test_table_text(self):
        assert derive_content_kind({"contains_table_text": True}) == "table_text"

    def test_table_caption_over_table_text(self):
        # table_text 优先级高于 table_caption
        assert derive_content_kind({"contains_table_text": True,
                                     "contains_table_caption": True}) == "table_text"

    def test_table_caption(self):
        assert derive_content_kind({"contains_table_caption": True}) == "table_caption"

    def test_figure_caption(self):
        assert derive_content_kind({"contains_figure_caption": True}) == "figure_caption"

    def test_image_related(self):
        assert derive_content_kind({"contains_image": True}) == "image_related"

    def test_references(self):
        assert derive_content_kind({"contains_references": True}) == "references"

    def test_metadata(self):
        assert derive_content_kind({"contains_metadata": True}) == "metadata"

    def test_default_body(self):
        assert derive_content_kind({}) == "body"

    def test_missing_fields_default_body(self):
        assert derive_content_kind({"some_other": True}) == "body"


# ============================================================
# 3. object_type 推导测试
# ============================================================

class TestDeriveObjectType:
    def test_table_text(self):
        assert derive_object_type("table_text") == "table"

    def test_table_caption(self):
        assert derive_object_type("table_caption") == "table"

    def test_figure_caption(self):
        assert derive_object_type("figure_caption") == "figure"

    def test_image_related(self):
        assert derive_object_type("image_related") == "image"

    def test_references(self):
        assert derive_object_type("references") == "references"

    def test_metadata(self):
        assert derive_object_type("metadata") == "metadata"

    def test_body(self):
        assert derive_object_type("body") == "body"

    def test_unknown_fallsback_body(self):
        assert derive_object_type("unknown_kind") == "body"


# ============================================================
# 4. object_id 推导测试
# ============================================================

class TestInferObjectId:
    def test_existing_object_id(self):
        chunk = {"doc_id": "doc1", "object_id": "obj-123", "source_block_ids": ["b1"]}
        assert infer_object_id(chunk, "table") == "obj-123"

    def test_existing_table_id(self):
        chunk = {"doc_id": "doc1", "table_id": "tbl-456"}
        assert infer_object_id(chunk, "table") == "tbl-456"

    def test_existing_figure_id(self):
        chunk = {"doc_id": "doc1", "figure_id": "fig-789"}
        assert infer_object_id(chunk, "figure") == "fig-789"

    def test_from_source_block_ids(self):
        chunk = {"doc_id": "doc1", "source_block_ids": ["b001", "b002"]}
        assert infer_object_id(chunk, "table") == "doc1::table::b001"

    def test_from_block_ids_fallback(self):
        chunk = {"doc_id": "doc2", "block_ids": ["bb01"]}
        assert infer_object_id(chunk, "figure") == "doc2::figure::bb01"

    def test_source_block_ids_priority_over_block_ids(self):
        chunk = {"doc_id": "doc3", "source_block_ids": ["s1"], "block_ids": ["b1"]}
        assert infer_object_id(chunk, "body") == "doc3::body::s1"

    def test_no_block_info_returns_empty(self):
        assert infer_object_id({"doc_id": "doc4"}, "body") == ""

    def test_empty_block_ids_returns_empty(self):
        assert infer_object_id({"doc_id": "doc5", "source_block_ids": []}, "table") == ""


# ============================================================
# 5. metadata_json 测试
# ============================================================

class TestBuildMetadataJson:
    def test_produces_valid_json(self):
        result = build_metadata_json({})
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_list_fields_serialized(self):
        chunk = {
            "section_path": ["Abstract", "Introduction"],
            "block_types": ["text", "table"],
            "source_block_ids": ["b1", "b2"],
            "evidence_types": ["result", "numeric"],
            "page_numbers": [1, 2],
            "reading_order_span": [0, 10],
            "bbox_span": [[0, 0, 100, 200]],
        }
        result = build_metadata_json(chunk)
        parsed = json.loads(result)
        assert parsed["section_path"] == ["Abstract", "Introduction"]
        assert parsed["block_types"] == ["text", "table"]
        assert parsed["evidence_types"] == ["result", "numeric"]
        assert parsed["page_numbers"] == [1, 2]

    def test_dict_fields_serialized(self):
        chunk = {"excluded_block_counts": {"noise": 3, "empty": 1}}
        result = build_metadata_json(chunk)
        parsed = json.loads(result)
        assert parsed["excluded_block_counts"] == {"noise": 3, "empty": 1}

    def test_chart_reserved_fields_present(self):
        result = build_metadata_json({})
        parsed = json.loads(result)
        reserved = [
            "table_id", "figure_id", "image_id", "object_id", "object_type",
            "table_caption", "table_markdown", "table_csv_path", "table_json_path",
            "figure_caption", "image_path", "image_hash", "ocr_text",
            "visual_summary", "asset_uri",
        ]
        for field in reserved:
            assert field in parsed, f"缺少预留字段: {field}"
            assert parsed[field] == "", f"预留字段 {field} 默认值应为空字符串"

    def test_default_values_for_missing_fields(self):
        result = build_metadata_json({})
        parsed = json.loads(result)
        assert parsed["section_path"] == []
        assert parsed["layout_columns"] == 1
        assert parsed["contains_references"] is False
        assert parsed["parser_stage"] == ""

    def test_source_block_metadata_summary(self):
        chunk = {
            "source_block_metadata": [
                {"block_id": "b1", "type": "text", "page": 1,
                 "bbox": [0, 0, 100, 200], "text_length": 500,
                 "full_text": "should be stripped"},
            ]
        }
        result = build_metadata_json(chunk)
        parsed = json.loads(result)
        sbm = parsed["source_block_metadata"]
        assert len(sbm) == 1
        assert "block_id" in sbm[0]
        assert "full_text" not in sbm[0]


# ============================================================
# 6. metadata_json 安全解析测试
# ============================================================

class TestSafeParseMetadataJson:
    def test_valid_json(self):
        result = _safe_parse_metadata_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_empty_string(self):
        assert _safe_parse_metadata_json("") == {}

    def test_none_input(self):
        assert _safe_parse_metadata_json(None) == {}

    def test_invalid_json(self):
        result = _safe_parse_metadata_json("{invalid")
        assert result == {"metadata_json_parse_error": True}

    def test_non_string_type(self):
        result = _safe_parse_metadata_json(123)
        assert result == {}

    def test_bm25_safe_parse_same_behavior(self):
        # BM25 和 Milvus 中的函数行为一致
        assert _bm25_safe_parse('{"a":1}') == {"a": 1}
        assert _bm25_safe_parse("bad") == {"metadata_json_parse_error": True}
        assert _bm25_safe_parse("") == {}


# ============================================================
# 7. BM25 retrieval_text 测试
# ============================================================

class TestBM25RetrievalText:
    def test_prefers_metadata_retrieval_text(self):
        chunk = RetrievedChunk(
            chunk_id="c1", doc_id="d1", source_file="f1",
            title="t1", section="s1", text="no keyword here",
            metadata={"retrieval_text": "table caption with specific keyword XYZ123"},
        )
        result = _retrieval_text(chunk)
        assert "XYZ123" in result
        assert "no keyword" not in result

    def test_fallback_when_no_retrieval_text(self):
        chunk = RetrievedChunk(
            chunk_id="c1", doc_id="d1", source_file="f1",
            title="My Title", section="Results", text="some body text",
            metadata={},
        )
        result = _retrieval_text(chunk)
        assert "My Title" in result
        assert "some body text" in result

    def test_bm25_index_uses_retrieval_text_for_tokenization(self):
        """BM25 索引时优先使用 retrieval_text 进行分词。"""
        chunk = RetrievedChunk(
            chunk_id="c1", doc_id="d1", source_file="f1",
            title="t1", section="s1", text="普通正文内容",
            metadata={"retrieval_text": "Figure 3 caption shows growth curve data"},
        )
        # _retrieval_text 返回 retrieval_text，BM25 索引对其分词
        rt = _retrieval_text(chunk)
        assert "Figure" in rt
        assert "growth curve" in rt


# ============================================================
# 8. search_params 测试
# ============================================================

class TestBuildSearchParams:
    def test_ivf_flat_uses_nprobe(self):
        config = RetrievalConfig(index_type="IVF_FLAT", nprobe=32)
        retriever = MilvusRetriever(config, MagicMock())
        params = retriever._build_search_params()
        assert params["metric_type"] == "COSINE"
        assert params["params"] == {"nprobe": 32}

    def test_hnsw_uses_ef(self):
        config = RetrievalConfig(index_type="HNSW", hnsw_ef=128)
        retriever = MilvusRetriever(config, MagicMock())
        params = retriever._build_search_params()
        assert params["metric_type"] == "COSINE"
        assert params["params"] == {"ef": 128}

    def test_default_ivf_flat(self):
        config = RetrievalConfig()
        retriever = MilvusRetriever(config, MagicMock())
        params = retriever._build_search_params()
        assert "nprobe" in params["params"]


# ============================================================
# 9. 兼容旧 chunks.jsonl 测试
# ============================================================

# ============================================================
# _safe_truncate 测试
# ============================================================

class TestSafeTruncate:
    def test_short_string_unchanged(self):
        result = _safe_truncate("hello", 100, "test_field", "chunk_001")
        assert result == "hello"

    def test_exact_length_unchanged(self):
        result = _safe_truncate("1234567890", 10, "test_field", "chunk_001")
        assert result == "1234567890"
        assert len(result) == 10

    def test_long_string_truncated(self):
        result = _safe_truncate("a" * 100, 50, "test_field", "chunk_001")
        assert len(result) == 50
        assert result == "a" * 50

    def test_text_uses_schema_limit_16384(self):
        long_text = "x" * 20000
        result = _safe_truncate(long_text, TEXT_VARCHAR_MAX_LENGTH, "text", "doc_001")
        assert len(result) == TEXT_VARCHAR_MAX_LENGTH

    def test_retrieval_text_uses_schema_limit_16384(self):
        long_text = "x" * 20000
        result = _safe_truncate(
            long_text, RETRIEVAL_TEXT_VARCHAR_MAX_LENGTH, "retrieval_text", "doc_001"
        )
        assert len(result) == RETRIEVAL_TEXT_VARCHAR_MAX_LENGTH

    def test_metadata_json_uses_schema_limit(self):
        long_json = "{" + "x" * 10000 + "}"
        result = _safe_truncate(
            long_json, METADATA_JSON_VARCHAR_MAX_LENGTH, "metadata_json", "doc_001"
        )
        assert len(result) == METADATA_JSON_VARCHAR_MAX_LENGTH

    def test_truncation_records_statistics(self):
        # 清理旧统计数据
        for key in _truncation_stats:
            _truncation_stats[key].clear()

        _safe_truncate("x" * 20000, TEXT_VARCHAR_MAX_LENGTH, "text", "doc_001")
        stats = _truncation_stats["text"]
        assert len(stats) == 1
        assert stats[0]["chunk_id"] == "doc_001"
        assert stats[0]["original_chars"] == 20000
        assert stats[0]["max_allowed"] == TEXT_VARCHAR_MAX_LENGTH

        # 清理
        for key in _truncation_stats:
            _truncation_stats[key].clear()

    def test_no_truncation_no_stats_recorded(self):
        for key in _truncation_stats:
            _truncation_stats[key].clear()

        _safe_truncate("short", TEXT_VARCHAR_MAX_LENGTH, "text", "doc_001")
        assert len(_truncation_stats["text"]) == 0

        for key in _truncation_stats:
            _truncation_stats[key].clear()


class TestBackwardCompatibility:
    def test_derive_content_kind_missing_all_fields(self):
        """旧 chunks.jsonl 缺所有 contains_* 字段，应返回 body。"""
        old_chunk = {"text": "some text", "doc_id": "d1"}
        assert derive_content_kind(old_chunk) == "body"

    def test_derive_object_type_from_body(self):
        assert derive_object_type("body") == "body"

    def test_build_metadata_json_empty_chunk(self):
        """空 chunk 不抛异常。"""
        result = build_metadata_json({})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_infer_object_id_no_doc_id(self):
        """没有 doc_id 和 block 信息时返回空字符串。"""
        assert infer_object_id({}, "body") == ""

    def test_build_metadata_json_source_block_metadata_not_list(self):
        """source_block_metadata 不是 list 时不崩溃。"""
        chunk = {"source_block_metadata": "not_a_list"}
        result = build_metadata_json(chunk)
        parsed = json.loads(result)
        assert parsed["source_block_metadata"] == []


# ============================================================
# 10. Config 新增字段测试
# ============================================================

class TestRetrievalConfigPhase12A:
    def test_default_values(self):
        config = RetrievalConfig()
        assert config.index_schema_version == "v2"
        assert config.index_type == "IVF_FLAT"
        assert config.nprobe == 16
        assert config.hnsw_ef == 64

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_INDEX_SCHEMA_VERSION", "v3")
        monkeypatch.setenv("RETRIEVAL_INDEX_TYPE", "HNSW")
        monkeypatch.setenv("RETRIEVAL_NPROBE", "32")
        monkeypatch.setenv("RETRIEVAL_HNSW_EF", "256")

        settings = Settings.from_env()
        assert settings.retrieval.index_schema_version == "v3"
        assert settings.retrieval.index_type == "HNSW"
        assert settings.retrieval.nprobe == 32
        assert settings.retrieval.hnsw_ef == 256
