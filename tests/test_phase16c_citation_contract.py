from __future__ import annotations

import pytest

from src.synbio_rag.application.generation_v2.citation_binder import CitationBinder
from src.synbio_rag.application.generation_v2.evidence_lifecycle_debug import (
    citation_candidates_debug,
    citation_output_debug,
)
from src.synbio_rag.application.generation_v2.models import (
    CitationCandidate,
    EvidenceCandidate,
    SupportItem,
)
from src.synbio_rag.domain.schemas import Citation


def _candidate(eid: str, text: str, **kwargs) -> EvidenceCandidate:
    defaults = {
        "evidence_id": eid,
        "chunk_id": f"{eid}_chunk",
        "doc_id": kwargs.pop("doc_id", "doc_001"),
        "source_file": kwargs.pop("source_file", "doc_001.pdf"),
        "title": kwargs.pop("title", "Test Paper"),
        "section": kwargs.pop("section", "Results"),
        "text": text,
        "page_start": kwargs.pop("page_start", None),
        "page_end": kwargs.pop("page_end", None),
        "vector_score": kwargs.pop("vector_score", 0.8),
        "bm25_score": kwargs.pop("bm25_score", 0.7),
        "rerank_score": kwargs.pop("rerank_score", 0.9),
        "fusion_score": kwargs.pop("fusion_score", 0.85),
        "metadata": kwargs.pop("metadata", {}),
        "features": kwargs.pop("features", {}),
        "reasons": kwargs.pop("reasons", []),
    }
    return EvidenceCandidate(**defaults)


def _support(eid: str, text: str, score: float = 0.9, **candidate_kwargs) -> SupportItem:
    return SupportItem(
        evidence_id=eid,
        candidate=_candidate(eid, text, **candidate_kwargs),
        support_score=score,
        reasons=candidate_kwargs.pop("reasons", ["test"]),
    )


# ── Test 1: selected_support → citation_candidates contract ──────────
def test_selected_support_to_candidate_contract() -> None:
    """citation_eligible selected_support must enter citation_candidates."""
    binder = CitationBinder()
    pack = [
        _support("E001", "Valid evidence text with enough characters.", rerank_score=0.9,
                  metadata={"rerank_rank": 1}),
        _support("E002", "Another valid chunk of text for testing.", rerank_score=0.8,
                  metadata={"rerank_rank": 2}),
        # E003: metadata-incomplete (no chunk_id)
        SupportItem(
            evidence_id="E003",
            candidate=_candidate("E003", "Some text but bad metadata.",
                                 chunk_id="", doc_id=""),
            support_score=0.3,
            reasons=[],
        ),
        # E004: text too short
        _support("E004", "Short.", rerank_score=0.5),
    ]

    candidates = binder.build_citation_candidates(pack, plan_mode="full")

    # E001, E002 should enter
    assert len([c for c in candidates if c.citation_eligible]) == 2
    # E003 should be dropped with metadata_missing
    e3 = next(c for c in candidates if c.evidence_id == "E003")
    assert e3.drop_reason == "metadata_missing"
    assert not e3.citation_eligible
    # E004 should be dropped with text_missing
    e4 = next(c for c in candidates if c.evidence_id == "E004")
    assert e4.drop_reason == "text_missing"
    assert not e4.citation_eligible


# ── Test 2: protected candidate has higher citation_priority ─────────
def test_protected_candidate_priority() -> None:
    """Protected selected_support candidate citation_priority > non-protected."""
    binder = CitationBinder()
    pack = [
        _support("E001", "Protected evidence text with enough chars.", rerank_score=0.9,
                  metadata={"rerank_rank": 1}),  # protected (rank <= 3)
        _support("E002", "Regular evidence text with enough chars too.", rerank_score=0.6,
                  metadata={"rerank_rank": 10}),  # not protected
    ]
    candidates = binder.build_citation_candidates(pack)

    protected = next(c for c in candidates if c.evidence_id == "E001")
    regular = next(c for c in candidates if c.evidence_id == "E002")
    assert protected.is_protected_seed
    assert not regular.is_protected_seed
    assert protected.citation_priority > regular.citation_priority, (
        f"Protected priority {protected.citation_priority} <= regular {regular.citation_priority}"
    )


# ── Test 3: no forced citation when marker is missing ─────────────────
def test_no_forced_citation_when_marker_missing() -> None:
    """Answer without [E#] marker must not force citation, but must record drop_reason."""
    binder = CitationBinder()
    pack = [
        _support("E001", "Valid evidence text with enough characters for testing."),
        _support("E002", "Another valid chunk of text for testing purposes."),
    ]
    answer = "This answer has no evidence markers at all."
    final_answer, citations, debug = binder.bind(answer, pack, plan_mode="full")

    assert len(citations) == 0
    assert len(debug["drop_reasons_by_evidence_id"]) == 2
    assert all(
        reason == "citation_marker_not_used"
        for reason in debug["drop_reasons_by_evidence_id"].values()
    )


# ── Test 4: partial mode does NOT silently drop candidates ────────────
def test_partial_mode_no_silent_drop() -> None:
    """Partial mode selected_support candidates must not be silently filtered."""
    binder = CitationBinder()
    pack = [
        _support("E001", "Cited evidence text with enough chars for test.", rerank_score=0.9,
                  metadata={"rerank_rank": 1}),
        _support("E002", "Uncited evidence text with enough chars for test.", rerank_score=0.7,
                  metadata={"rerank_rank": 3}),
    ]
    answer = "Partial answer citing only [E001]."
    final_answer, citations, debug = binder.bind(answer, pack, plan_mode="partial")

    # E001 should be cited
    assert len(citations) == 1
    assert citations[0].chunk_id == "E001_chunk"

    # E002 NOT cited, but has an explicit drop_reason (NOT partial_mode_filtered)
    assert "E002" in debug["drop_reasons_by_evidence_id"]
    reason = debug["drop_reasons_by_evidence_id"]["E002"]
    assert reason == "citation_marker_not_used", (
        f"Expected citation_marker_not_used, got {reason}"
    )

    # Verify via evidence_lifecycle_debug too
    cit_debug = {
        "ordered_evidence_ids": debug["ordered_evidence_ids"],
        "invalid_evidence_ids": debug["invalid_evidence_ids"],
    }
    output_debug = citation_output_debug(
        support_pack=pack, citations=citations,
        citation_debug=cit_debug, plan_mode="partial",
    )
    assert output_debug["partial_mode"] is True
    assert output_debug["drop_reasons"].get("E002_chunk") == "citation_marker_not_used"


# ── Test 5: comparison branch tracking ───────────────────────────────
def test_comparison_branch_trace() -> None:
    """Multi expected_doc scenario records each branch candidate/output state."""
    binder = CitationBinder()
    pack = [
        _support("E001", "Branch A evidence text with enough characters here.",
                  doc_id="doc_A", rerank_score=0.9, metadata={"rerank_rank": 1}),
        _support("E002", "Branch B evidence text with sufficient length for testing.",
                  doc_id="doc_B", rerank_score=0.8, metadata={"rerank_rank": 2}),
    ]
    answer = "Comparison answer citing only branch A: [E001]."
    final_answer, citations, debug = binder.bind(answer, pack, plan_mode="partial")

    # Branch A (doc_A) is cited
    cited_docs = {c.doc_id for c in citations}
    assert "doc_A" in cited_docs
    # Branch B (doc_B) is NOT cited
    assert "doc_B" not in cited_docs

    # Both branches have candidates
    cc_docs = {c.doc_id for c in binder.build_citation_candidates(pack)}
    assert "doc_A" in cc_docs
    assert "doc_B" in cc_docs

    # E002 (branch B) has drop_reason recorded
    assert debug["drop_reasons_by_evidence_id"].get("E002") == "citation_marker_not_used"


# ── Test 6: citation limit unchanged ──────────────────────────────────
def test_citation_limit_unchanged() -> None:
    """Citation output count does not exceed the number of [E#] markers in answer."""
    binder = CitationBinder()
    pack = [
        _support(f"E{i:03d}", f"Evidence chunk number {i} with sufficient text length for testing.",
                  rerank_score=0.9 - i * 0.01, metadata={"rerank_rank": i})
        for i in range(1, 11)
    ]
    # Answer only references 3 markers
    answer = "Answer with [E001], [E003], and [E005] only."
    final_answer, citations, debug = binder.bind(answer, pack)

    # Citations should be exactly 3 (only the markers in answer)
    assert len(citations) == 3, f"Expected 3 citations, got {len(citations)}"
    cited_ids = {c.chunk_id for c in citations}
    assert cited_ids == {"E001_chunk", "E003_chunk", "E005_chunk"}


# ── Test 7: behavior compatibility for unaffected cases ───────────────
def test_behavior_compatibility_unaffected_cases() -> None:
    """Non-partial, non-protected seed case behavior remains unchanged."""
    binder = CitationBinder()
    pack = [
        _support("E001", "Valid evidence text one with enough characters for testing purposes.",
                  rerank_score=0.8, metadata={"rerank_rank": 5}),
        _support("E002", "Valid evidence text two with enough characters for testing too.",
                  rerank_score=0.7, metadata={"rerank_rank": 6}),
    ]
    answer = "Answer citing [E001] and [E002]."
    final_answer, citations, debug = binder.bind(answer, pack, plan_mode="full")

    assert "[1]" in final_answer
    assert "[2]" in final_answer
    assert len(citations) == 2
    assert citations[0].chunk_id == "E001_chunk"
    assert citations[1].chunk_id == "E002_chunk"
    assert len(debug["drop_reasons_by_evidence_id"]) == 0
    assert debug.get("plan_mode") == "full"


# ── Test 8: citation_candidates_debug with CitationCandidate objects ──
def test_citation_candidates_debug_with_candidates() -> None:
    """citation_candidates_debug records per-candidate drop_reasons and eligibility."""
    binder = CitationBinder()
    pack = [
        _support("E001", "Valid evidence with sufficient text length here.", rerank_score=0.9,
                  metadata={"rerank_rank": 1}),
    ]
    # E003: bad metadata
    pack.append(SupportItem(
        evidence_id="E003",
        candidate=_candidate("E003", "Some text but incomplete metadata.", chunk_id="", doc_id=""),
        support_score=0.3,
        reasons=[],
    ))

    candidates = binder.build_citation_candidates(pack)
    debug = citation_candidates_debug(pack, citation_candidates=candidates)

    assert debug["input_count"] == 2
    assert debug["output_count"] == 2
    assert debug["citation_eligible_count"] == 1
    assert "protected_seed_count" in debug
    assert "drop_reasons" in debug
    # E003 should have metadata_missing drop_reason
    assert any("E003_chunk" in k or "metadata_missing" in str(v)
               for k, v in debug.get("drop_reasons", {}).items())


# ── Test 9: citation output debug partial_mode track ─────────────────
def test_citation_output_debug_partial_mode_tracking() -> None:
    """citation_output_debug tracks partial_mode separately from drop_reason."""
    c1 = _candidate("E001", "Evidence text one with enough characters for testing purposes.")
    c2 = _candidate("E002", "Evidence text two with enough characters for testing purposes.",
                     doc_id="doc_002", source_file="doc_002.pdf")
    pack = [
        SupportItem(evidence_id="E001", candidate=c1, support_score=0.9, reasons=[]),
        SupportItem(evidence_id="E002", candidate=c2, support_score=0.7, reasons=[]),
    ]
    citations = [
        Citation(chunk_id="E001_chunk", doc_id="doc_001", title="", source_file="doc_001.pdf",
                 section="Results", page_start=None, page_end=None, score=0.9, quote="Evidence text one..."),
    ]
    cit_debug = {"ordered_evidence_ids": ["E001"], "invalid_evidence_ids": []}

    # partial mode
    debug = citation_output_debug(
        support_pack=pack, citations=citations,
        citation_debug=cit_debug, plan_mode="partial",
    )
    assert debug["partial_mode"] is True
    assert "E002_chunk" in debug["partial_mode_uncited_chunk_ids"]
    # drop_reason is still citation_marker_not_used, NOT partial_mode_filtered
    assert debug["drop_reasons"]["E002_chunk"] == "citation_marker_not_used"

    # full mode
    debug_full = citation_output_debug(
        support_pack=pack, citations=citations,
        citation_debug=cit_debug, plan_mode="full",
    )
    assert debug_full["partial_mode"] is False
    assert debug_full["partial_mode_uncited_chunk_ids"] == []
    assert debug_full["drop_reasons"]["E002_chunk"] == "citation_marker_not_used"
