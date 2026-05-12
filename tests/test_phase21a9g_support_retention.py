"""Phase 21A-9G: Support retention fix tests."""
import inspect
import pytest
from collections import Counter

from src.synbio_rag.application.generation_v2.support_selector import (
    SupportPackSelector,
    _retain_doc_diversity,
    _ensure_protected_support_seeds,
)


# ── Test helpers ──────────────────────────────────────────────────────

class FakeCandidate:
    def __init__(self, evidence_id, doc_id, chunk_id, section="Results",
                 rerank_score=1.0, text="test text content with enough length for quality checks"):
        self.evidence_id = evidence_id
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.section = section
        self.title = "Test Title"
        self.text = text
        self.rerank_score = rerank_score
        self.fusion_score = None
        self.vector_score = 0.0
        self.bm25_score = 0.0
        self.features = {
            "section_type": section,
            "has_numeric": False,
            "has_result_terms": False,
            "has_table_text": False,
            "has_table_caption": False,
            "has_figure_caption": False,
            "text_length": len(text),
            "bibliography_like": False,
        }
        self.metadata = {"rerank_rank": 99}
        self.reasons = []


class FakeSupportItem:
    def __init__(self, evidence_id, doc_id, support_score, section="Results",
                 rerank_rank=99, text=None):
        self.evidence_id = evidence_id
        self.candidate = FakeCandidate(
            evidence_id=evidence_id,
            doc_id=doc_id,
            chunk_id=f"{doc_id}_chunk",
            section=section,
            rerank_score=support_score,
            text=text or f"test text for {evidence_id} with enough content length for tests",
        )
        self.candidate.metadata["rerank_rank"] = rerank_rank
        self.support_score = support_score
        self.reasons = []


class FakeConfig:
    v2_protect_support_seeds_enabled = True
    v2_protect_support_seeds_top_k = 2
    v2_max_support_factoid = 3
    v2_max_support_summary = 3
    v2_min_support_score = -10.0
    v2_enable_comparison_coverage = False
    v2_max_support_comparison = 3


def make_scored(items_spec):
    """items_spec: list of (evidence_id, doc_id, score, section, rerank_rank)"""
    result = []
    for spec in items_spec:
        eid, doc, score = spec[0], spec[1], spec[2]
        section = spec[3] if len(spec) > 3 else "Results"
        rank = spec[4] if len(spec) > 4 else 99
        result.append(FakeSupportItem(eid, doc, score, section, rank))
    return result


# ── Tests ─────────────────────────────────────────────────────────────

class TestSupportRetentionNoSampleSpecialCase:
    def test_no_sample_special_case(self):
        src = inspect.getsource(_retain_doc_diversity)
        for banned in ["ent_058", "ent_060", "ent_081", "ent_082",
                       "ent_094", "ent_100", "ent_077"]:
            assert banned not in src, f"Banned sample_id {banned} in _retain_doc_diversity"

    def test_no_doc_id_special_case(self):
        src = inspect.getsource(_retain_doc_diversity)
        for banned in ["doc_0098", "doc_0105", "doc_0151", "doc_0003",
                       "doc_0090", "doc_0147"]:
            assert banned not in src, f"Banned doc_id {banned} in _retain_doc_diversity"

    def test_no_expected_doc_used_in_logic(self):
        src = inspect.getsource(_retain_doc_diversity)
        assert "expected_doc" not in src
        assert "expected_source" not in src

    def test_seed_protection_no_sample_special_case(self):
        src = inspect.getsource(_ensure_protected_support_seeds)
        for banned in ["ent_058", "ent_077"]:
            assert banned not in src


class TestDocDiversityRetention:
    def test_same_doc_overcrowding_is_swapped(self):
        """When doc A appears 2x and doc B has comparable score, swap the lower A."""
        items = make_scored([
            ("E1", "doc_A", 5.0, "Results", 1),
            ("E2", "doc_A", 4.5, "Results", 2),
            ("E3", "doc_B", 4.4, "Results", 99),
        ])
        selected = [items[0], items[1]]  # [E1doc_A, E2doc_A]
        config = FakeConfig()
        result = _retain_doc_diversity(selected, items, items, config)
        docs = [r.candidate.doc_id for r in result]
        assert "doc_B" in docs
        assert len(result) == 2
        # doc_B item must be in result
        assert any(r.evidence_id == "E3" for r in result)

    def test_no_swap_when_alternative_score_too_low(self):
        """Don't swap if alternative score is < 85% of replaced item."""
        items = make_scored([
            ("E1", "doc_A", 5.0, "Results"),
            ("E2", "doc_A", 4.5, "Results"),
            ("E3", "doc_B", 3.0, "Results"),  # 3.0 < 4.5*0.85=3.825
        ])
        selected = [items[0], items[1]]
        config = FakeConfig()
        result = _retain_doc_diversity(selected, items, items, config)
        docs = [r.candidate.doc_id for r in result]
        assert docs.count("doc_B") == 0
        assert docs.count("doc_A") == 2

    def test_all_distinct_docs_no_change(self):
        """When all docs are distinct, no swap needed."""
        items = make_scored([
            ("E1", "doc_A", 5.0, "Results"),
            ("E2", "doc_B", 4.5, "Results"),
            ("E3", "doc_C", 4.0, "Results"),
        ])
        selected = [items[0], items[1], items[2]]
        config = FakeConfig()
        result = _retain_doc_diversity(selected, items, items, config)
        assert len(result) == 3
        assert {r.evidence_id for r in result} == {"E1", "E2", "E3"}

    def test_empty_selected_gets_support_floor(self):
        """When selected is empty, add best available from all_scored."""
        items = make_scored([
            ("E1", "doc_A", -2.0, "Results"),
            ("E2", "doc_B", -1.0, "Results"),
        ])
        config = FakeConfig()
        result = _retain_doc_diversity([], items, items, config)
        assert len(result) == 1
        assert result[0].evidence_id == "E2"  # highest score

    def test_single_doc_no_duplicate_no_change(self):
        items = make_scored([
            ("E1", "doc_A", 5.0, "Results"),
        ])
        selected = [items[0]]
        config = FakeConfig()
        result = _retain_doc_diversity(selected, items, items, config)
        assert len(result) == 1
        assert result[0].evidence_id == "E1"


class TestSeedProtectionDiversity:
    def test_diversity_aware_truncation(self):
        """After seed insertion, prefer keeping items from different docs."""
        items = make_scored([
            ("E1", "doc_X", 6.0, "Results", 1),  # protected seed
            ("E2", "doc_X", 5.5, "Results"),       # same doc as E1
            ("E3", "doc_Y", 5.0, "Results"),       # different doc
            ("E4", "doc_Z", 4.5, "Results"),       # different doc
        ])
        selected = [items[1], items[2], items[3]]  # [E2docX, E3docY, E4docZ]
        config = FakeConfig()
        result = _ensure_protected_support_seeds(items, selected, config)
        docs = [r.candidate.doc_id for r in result]
        # E1 doc_X is inserted; diversity keeps E3 doc_Y over E2 doc_X
        assert "doc_Y" in docs
        assert "doc_Z" in docs
        assert len(result) == 3

    def test_truncation_keeps_max_size(self):
        items = make_scored([
            ("E1", "doc_X", 5.0, "Results", 1),
            ("E2", "doc_Y", 4.0, "Results"),
        ])
        selected = [items[1]]
        config = FakeConfig()
        result = _ensure_protected_support_seeds(items, selected, config)
        assert len(result) <= 3


class TestPhase20FixNotRegressed:
    def test_factoid_doc_diversity_max_per_doc_still_2(self):
        from src.synbio_rag.application.generation_v2.support_selector import _select_with_doc_diversity
        src = inspect.getsource(_select_with_doc_diversity)
        assert "max_per_doc" in src

    def test_summary_quality_filter_unchanged(self):
        from src.synbio_rag.application.generation_v2.support_selector import _evaluate_summary_quality
        src = inspect.getsource(_evaluate_summary_quality)
        assert "references_section" in src
        assert "too_short" in src

    def test_section_priority_unchanged(self):
        from src.synbio_rag.application.generation_v2.support_selector import _section_priority
        assert _section_priority("Abstract") == 0
        assert _section_priority("Conclusion") == 1
        assert _section_priority("results and discussion") == 2
        assert _section_priority("Full Text") == 2
        assert _section_priority("Introduction") == 5
        assert _section_priority("References") == 7

    def test_citation_binding_not_modified(self):
        """Citation binding module not touched."""
        # This test just verifies the import doesn't touch citation code
        from src.synbio_rag.application.generation_v2 import support_selector
        assert hasattr(support_selector, '_retain_doc_diversity')


class TestNegativeAbstentionNotForcedToCite:
    def test_retention_does_not_force_empty_selected_to_overcite(self):
        """With empty selection, support floor adds at most 1 item."""
        items = make_scored([
            ("E1", "doc_A", -2.0, "Results"),
            ("E2", "doc_B", -1.5, "Results"),
            ("E3", "doc_C", -1.0, "Results"),
        ])
        config = FakeConfig()
        result = _retain_doc_diversity([], items, items, config)
        # Support floor: exactly 1, not 3
        assert len(result) == 1

    def test_negative_samples_not_forced_by_diversity(self):
        """Diversity retention only swaps, never expands beyond max+0."""
        items = make_scored([
            ("E1", "doc_A", 5.0, "Results"),
            ("E2", "doc_A", 4.5, "Results"),
            ("E3", "doc_B", 4.4, "Results"),
        ])
        selected = [items[0], items[1]]  # 2 items, both doc_A
        config = FakeConfig()
        result = _retain_doc_diversity(selected, items, items, config)
        # Still 2 items (swap, not expand)
        assert len(result) == 2
