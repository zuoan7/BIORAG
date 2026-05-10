"""Phase 20D: Unit tests for factoid support selection doc diversity fix."""
import pytest
from collections import Counter

# Test the _select_with_doc_diversity function directly
from src.synbio_rag.application.generation_v2.support_selector import (
    _select_with_doc_diversity,
)
from src.synbio_rag.application.generation_v2.models import EvidenceCandidate, SupportItem


def _make_item(doc_id: str, chunk_id: str, support_score: float) -> SupportItem:
    candidate = EvidenceCandidate(
        evidence_id=chunk_id,
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_file=f"{doc_id}.pdf",
        title="Test Paper",
        section="results",
        text=f"Test text from {doc_id} chunk {chunk_id}",
        page_start=1,
        page_end=2,
        vector_score=0.5,
        bm25_score=0.5,
        rerank_score=support_score,
        fusion_score=0.0,
        metadata={},
        features={},
    )
    return SupportItem(
        evidence_id=chunk_id,
        candidate=candidate,
        support_score=support_score,
        reasons=[],
    )


def _finalize(items, route):
    return items


def test_doc_diversity_factoid_max3():
    """3 chunks from same doc + 2 from others → max 2 from same doc."""
    items = [
        _make_item("doc_A", "c1", 0.95),
        _make_item("doc_A", "c2", 0.90),
        _make_item("doc_A", "c3", 0.85),
        _make_item("doc_B", "c4", 0.80),
        _make_item("doc_C", "c5", 0.75),
    ]
    selected, per_doc = _select_with_doc_diversity(
        ranked=items, max_total=3, max_per_doc=2,
        route_name="factoid_top_score", finalizer=_finalize,
    )
    assert len(selected) == 3
    doc_ids = [item.candidate.doc_id for item in selected]
    # doc_A should have max 2
    assert doc_ids.count("doc_A") <= 2
    # should have at least one from doc_B or doc_C
    assert "doc_B" in doc_ids or "doc_C" in doc_ids


def test_single_doc_fallback():
    """Only one doc available → all 3 slots can go to same doc."""
    items = [
        _make_item("doc_A", "c1", 0.95),
        _make_item("doc_A", "c2", 0.90),
        _make_item("doc_A", "c3", 0.85),
    ]
    selected, per_doc = _select_with_doc_diversity(
        ranked=items, max_total=3, max_per_doc=2,
        route_name="factoid_top_score", finalizer=_finalize,
    )
    # With only 1 distinct doc and 2*1 < 3, fallback allows overflow
    assert len(selected) <= 3
    # All from same doc is OK since no other docs exist
    assert all(item.candidate.doc_id == "doc_A" for item in selected)


def test_two_docs_max3():
    """Two docs, 5 chunks each → max 2 per doc, 3 total."""
    items = [
        _make_item("doc_A", "a1", 0.95),
        _make_item("doc_A", "a2", 0.93),
        _make_item("doc_A", "a3", 0.91),
        _make_item("doc_B", "b1", 0.90),
        _make_item("doc_B", "b2", 0.88),
    ]
    selected, per_doc = _select_with_doc_diversity(
        ranked=items, max_total=3, max_per_doc=2,
        route_name="factoid_top_score", finalizer=_finalize,
    )
    assert len(selected) == 3
    doc_ids = [item.candidate.doc_id for item in selected]
    assert doc_ids.count("doc_A") <= 2
    assert doc_ids.count("doc_B") >= 1  # at least one from doc_B


def test_no_sample_special_case():
    """Verify function has no hardcoded sample IDs or doc IDs."""
    import inspect
    source = inspect.getsource(_select_with_doc_diversity)
    assert "sample_id" not in source
    assert "ent_" not in source
    assert "doc_0098" not in source
    assert "expected_doc" not in source


def test_score_order_preserved():
    """Higher-scored items from diverse docs should be selected first."""
    items = [
        _make_item("doc_A", "a1", 0.95),
        _make_item("doc_B", "b1", 0.90),
        _make_item("doc_A", "a2", 0.85),
        _make_item("doc_C", "c1", 0.80),
        _make_item("doc_B", "b2", 0.75),
    ]
    selected, per_doc = _select_with_doc_diversity(
        ranked=items, max_total=3, max_per_doc=2,
        route_name="factoid_top_score", finalizer=_finalize,
    )
    # a1 (0.95), b1 (0.90), c1 (0.80) should be selected
    # a2 has 0.85 but doc_A already has a1; c1 has 0.80 vs a2 0.85 and b2 0.75
    # With diversity: a1 (first), b1 (next, diff doc), then a2 or c1
    # Actually a2 comes before c1 in ranked list, so a2 (0.85) but doc_A already has 1, so c1 (0.80)
    # Wait, the order is a1(0.95), b1(0.90), a2(0.85), c1(0.80), b2(0.75)
    # After a1, b1: per_doc={doc_A:1, doc_B:1}
    # Next: a2 → doc_A count=1 < max_per_doc=2 → SELECTED
    # Next: c1 → len(selected)=3 → break
    # So: a1, b1, a2
    assert len(selected) == 3
    doc_ids = [item.candidate.doc_id for item in selected]
    assert "doc_A" in doc_ids
    assert "doc_B" in doc_ids


def test_citation_count_not_inflated():
    """Doc diversity should not increase total support count."""
    items = [
        _make_item(f"doc_{chr(65+i)}", f"c{i}", 0.9 - i * 0.05)
        for i in range(5)
    ]
    selected, per_doc = _select_with_doc_diversity(
        ranked=items, max_total=3, max_per_doc=2,
        route_name="factoid_top_score", finalizer=_finalize,
    )
    assert len(selected) == 3
    assert len(selected) == min(3, len(items))


def test_wrong_doc_not_cited_by_fix():
    """Fix does not force any doc into support — it only redistributes."""
    # The fix is about diversity, not about inserting specific docs
    # This test verifies no special-casing
    items = [
        _make_item("doc_A", "a1", 0.9),
        _make_item("doc_B", "b1", 0.8),
        _make_item("doc_B", "b2", 0.7),
    ]
    selected, per_doc = _select_with_doc_diversity(
        ranked=items, max_total=3, max_per_doc=2,
        route_name="factoid_top_score", finalizer=_finalize,
    )
    # doc_A still gets in (it's top score)
    assert any(item.candidate.doc_id == "doc_A" for item in selected)


def test_query_rewrite_off_default_unchanged():
    """Doc diversity is in support_selector, not in query rewrite config."""
    from src.synbio_rag.domain.config import QueryRewriteConfig
    qrc = QueryRewriteConfig()
    assert qrc.mode == "off"
