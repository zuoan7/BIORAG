"""Phase 20G: Summary quality minimal fix tests."""
import pytest
from src.synbio_rag.application.generation_v2.support_selector import (
    _evaluate_summary_quality,
    _section_priority,
    _summary_rank_key,
    _select_with_doc_diversity,
)
from src.synbio_rag.application.generation_v2.models import EvidenceCandidate, SupportItem
from collections import Counter


def _make_item(section: str, text: str = "Test content with enough length for quality evaluation purposes.",
               support_score: float = 0.8, features: dict = None,
               has_query_overlap: bool = True, reasons: list = None) -> SupportItem:
    if reasons is None:
        reasons = []
    if has_query_overlap:
        reasons.append("query_overlap")
    candidate = EvidenceCandidate(
        evidence_id="test_eid", chunk_id="test_cid", doc_id="test_doc",
        source_file="test.pdf", title="Test",
        section=section, text=text, page_start=1, page_end=2,
        vector_score=0.5, bm25_score=0.5, rerank_score=support_score,
        fusion_score=0.0, metadata={},
        features=features or {
            "text_length": len(text),
            "has_result_terms": "result" in text.lower(),
            "has_numeric": bool(__import__('re').search(r'\d+', text)),
            "has_table_text": False,
            "has_table_caption": False,
            "has_figure_caption": False,
        },
    )
    return SupportItem(evidence_id="test_eid", candidate=candidate, support_score=support_score, reasons=reasons)


def test_title_section_not_hard_dropped():
    """section='Title' should NOT be hard-dropped by _evaluate_summary_quality."""
    item = _make_item(section="Title", text="Engineering E. coli for 6'-sialyllactose biosynthesis pathway optimization.")
    eligible, reasons = _evaluate_summary_quality(item, rank_index=0, top_score_count=4)
    assert eligible, f"Title section should pass quality filter, got: {reasons}"


def test_title_still_not_forced_selected():
    """Title chunk just passes filter, not forced selected — rank key still applies."""
    item = _make_item(section="Title", support_score=0.3)
    eligible, _ = _evaluate_summary_quality(item, rank_index=0, top_score_count=4)
    assert eligible
    rk = _summary_rank_key(item)
    # Title should not be in _section_priority's 0-5 range
    sp = _section_priority("Title")
    assert sp >= 5, f"Title section_priority should be >=5 (background level), got {sp}"


def test_full_text_has_reasonable_section_priority():
    """_section_priority('Full Text') should return 2."""
    assert _section_priority("Full Text") == 2


def test_full_text_priority_does_not_override_results():
    """Results/Discussion/Abstract still have their original priorities."""
    assert _section_priority("Abstract") == 0
    assert _section_priority("Results and Discussion") == 2
    assert _section_priority("Results") == 3
    assert _section_priority("Discussion") == 4
    assert _section_priority("Conclusion") == 1


def test_summary_rank_key_not_rewritten():
    """Verify summary_rank_key structure unchanged."""
    item = _make_item(section="Results", support_score=0.85,
                      features={"text_length": 200, "has_result_terms": True, "has_numeric": True,
                                "has_table_text": False, "has_table_caption": False, "has_figure_caption": False})
    rk = _summary_rank_key(item)
    assert len(rk) == 4
    assert isinstance(rk[0], int)  # section_priority
    assert isinstance(rk[1], int)  # has_result_terms flag
    assert isinstance(rk[2], int)  # has_numeric flag
    assert isinstance(rk[3], float)  # -support_score


def test_factoid_selector_unaffected():
    """_select_with_doc_diversity still exists and works."""
    items = [
        _make_item(section="Results", support_score=0.9, has_query_overlap=True,
                   features={"has_result_terms": True, "has_numeric": True, "has_table_text": False,
                             "has_table_caption": False, "has_figure_caption": False, "text_length": 100}),
        _make_item(section="Results", support_score=0.8, has_query_overlap=True,
                   features={"has_result_terms": True, "has_numeric": False, "has_table_text": False,
                             "has_table_caption": False, "has_figure_caption": False, "text_length": 100}),
        _make_item(section="Discussion", support_score=0.7, has_query_overlap=True,
                   features={"has_result_terms": True, "has_numeric": True, "has_table_text": False,
                             "has_table_caption": False, "has_figure_caption": False, "text_length": 100}),
    ]
    # Change doc_ids to test diversity
    items[0].candidate.doc_id = "doc_A"
    items[1].candidate.doc_id = "doc_A"
    items[2].candidate.doc_id = "doc_B"
    selected, per_doc = _select_with_doc_diversity(
        ranked=items, max_total=3, max_per_doc=2,
        route_name="factoid_test", finalizer=lambda x, r: x,
    )
    assert len(selected) >= 2
    assert per_doc["doc_A"] <= 2


def test_no_sample_special_case():
    """Source code must not contain sample IDs or doc IDs."""
    import inspect
    src = inspect.getsource(_evaluate_summary_quality) + inspect.getsource(_section_priority)
    for banned in ["ent_005", "ent_058", "ent_081", "doc_0009", "doc_0098", "doc_0151"]:
        assert banned not in src, f"Found banned string {banned} in source"


def test_citation_capacity_not_increased():
    """Support capacity unchanged."""
    from src.synbio_rag.domain.config import GenerationConfig
    c = GenerationConfig()
    assert c.v2_max_support_summary == 5
    assert c.v2_max_support_factoid == 3
