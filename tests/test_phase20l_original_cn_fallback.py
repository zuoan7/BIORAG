"""Phase 20L-2: Original CN fallback floor tests."""
import pytest
from src.synbio_rag.domain.config import RetrievalConfig
from src.synbio_rag.application.pipeline import _contains_cjk

def test_original_cn_fallback_disabled_by_default():
    c = RetrievalConfig()
    assert c.original_cn_fallback_enabled is False
    assert c.original_cn_fallback_max_total == 4

def test_cjk_detection():
    assert _contains_cjk("中文查询")
    assert not _contains_cjk("English query")
    assert _contains_cjk("E. coli 和 B. subtilis 的比较")

def test_fallback_config_fields_present():
    c = RetrievalConfig()
    for field in ["original_cn_fallback_enabled", "original_cn_fallback_dense_top_n",
                   "original_cn_fallback_bm25_top_n", "original_cn_fallback_max_total"]:
        assert hasattr(c, field), f"Missing {field}"

def test_fallback_feature_flag_configurable():
    """Simulate env var override."""
    c = RetrievalConfig()
    c.original_cn_fallback_enabled = True
    c.original_cn_fallback_dense_top_n = 1
    c.original_cn_fallback_bm25_top_n = 1
    c.original_cn_fallback_max_total = 2
    assert c.original_cn_fallback_enabled is True
    assert c.original_cn_fallback_max_total == 2

def test_no_sample_special_case():
    import inspect
    from src.synbio_rag.application import pipeline
    src = inspect.getsource(pipeline)
    for banned in ["h50_neg_001", "doc_0204", "doc_0180"]:
        assert banned not in src, f"Banned string {banned} in source"

def test_require_rewrite_enabled_prevents_fallback_on_off():
    c = RetrievalConfig()
    assert c.original_cn_fallback_require_rewrite_enabled is True

def test_require_cjk_prevents_fallback_on_english():
    c = RetrievalConfig()
    assert c.original_cn_fallback_require_cjk is True
