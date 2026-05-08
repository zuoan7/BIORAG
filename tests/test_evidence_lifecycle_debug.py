from __future__ import annotations

from src.synbio_rag.application.generation_v2.citation_binder import CitationBinder
from src.synbio_rag.application.generation_v2.evidence_lifecycle_debug import (
    ALLOWED_DROP_REASONS,
    citation_output_debug,
    selected_support_debug,
)
from src.synbio_rag.application.generation_v2.models import EvidenceCandidate, SupportItem
from src.synbio_rag.application.generation_v2.support_selector import SupportPackSelector
from src.synbio_rag.domain.config import GenerationConfig
from src.synbio_rag.domain.schemas import QueryAnalysis, QueryIntent


def _analysis(intent: QueryIntent = QueryIntent.FACTOID) -> QueryAnalysis:
    return QueryAnalysis(intent=intent, requires_external_tools=False, search_limit=10, rerank_top_k=5)


def _candidate(
    evidence_id: str,
    *,
    chunk_id: str | None = None,
    doc_id: str = "doc1",
    text: str = "Results showed production increased to 10 g/L.",
    rerank: float = 0.8,
    metadata: dict | None = None,
) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        chunk_id=chunk_id or f"{evidence_id.lower()}_chunk",
        doc_id=doc_id,
        source_file=f"{doc_id}.pdf",
        title=f"title-{doc_id}",
        section="Results",
        text=text,
        page_start=None,
        page_end=None,
        vector_score=0.0,
        bm25_score=0.0,
        rerank_score=rerank,
        fusion_score=0.0,
        metadata=metadata or {},
        features={"section_type": "results", "has_result_terms": True, "text_length": len(text)},
        reasons=["seed_chunk"],
    )


def test_debug_instrumentation_does_not_change_support_or_citation_behavior() -> None:
    candidates = [_candidate("E1"), _candidate("E2", doc_id="doc2")]
    selector = SupportPackSelector()
    config = GenerationConfig(v2_max_support_factoid=1, v2_protect_support_seeds_enabled=False)

    selected = selector.select("production result", _analysis(), candidates, config)
    selected_ids_before_debug = [item.evidence_id for item in selected]

    lifecycle = selected_support_debug(
        candidates=candidates,
        support_pack=selected,
        selector_debug=selector.last_selection_debug,
        answer_mode="full",
        plan_mode="full",
        support_pack_size=len(selected),
    )

    binder = CitationBinder()
    answer, citations, citation_debug = binder.bind("Answer [E1].", selected)
    citation_lifecycle = citation_output_debug(
        support_pack=selected,
        citations=citations,
        citation_debug=citation_debug,
        plan_mode="full",
    )

    assert [item.evidence_id for item in selected] == selected_ids_before_debug
    assert answer == "Answer [1]."
    assert [citation.chunk_id for citation in citations] == [selected[0].candidate.chunk_id]
    assert lifecycle["output_count"] == len(selected)
    assert citation_lifecycle["output_count"] == len(citations)


def test_drop_reason_enum_consistency() -> None:
    candidates = [_candidate("E1"), _candidate("E2", doc_id="doc2")]
    selected = [SupportItem("E1", candidates[0], 0.8, ["factoid_top_score"])]
    debug = selected_support_debug(
        candidates=candidates,
        support_pack=selected,
        selector_debug={"drop_reasons_by_evidence_id": {"E2": "support_pack_size_limit"}},
        answer_mode="full",
        plan_mode="full",
        support_pack_size=1,
    )

    assert set(debug["drop_reasons"].values()).issubset(ALLOWED_DROP_REASONS)


def test_protected_seed_debug_counts_kept_and_dropped() -> None:
    candidates = [
        _candidate("E1", metadata={"rerank_rank": 1}),
        _candidate("E2", doc_id="doc2", metadata={"rerank_rank": 2}),
    ]
    selected = [SupportItem("E1", candidates[0], 0.8, ["factoid_top_score"])]

    debug = selected_support_debug(
        candidates=candidates,
        support_pack=selected,
        selector_debug={"drop_reasons_by_evidence_id": {"E2": "support_pack_size_limit"}},
        answer_mode="full",
        plan_mode="full",
        support_pack_size=1,
    )

    assert debug["protected_seed_kept_count"] == 1
    assert debug["protected_seed_dropped_count"] == 1
    assert debug["protected_seed_dropped_chunk_ids"] == ["e2_chunk"]


def test_comparison_branch_trace_inputs_can_be_checked_per_expected_doc() -> None:
    candidates = [_candidate("E1", doc_id="docA"), _candidate("E2", doc_id="docB")]
    selected = [SupportItem("E1", candidates[0], 0.8, ["comparison_selection"])]
    selected_docs = {item.candidate.doc_id for item in selected}

    branch_rows = [
        {"branch_expected_doc_id": doc_id, "branch_doc_in_selected_support": doc_id in selected_docs}
        for doc_id in ["docA", "docB"]
    ]

    assert branch_rows == [
        {"branch_expected_doc_id": "docA", "branch_doc_in_selected_support": True},
        {"branch_expected_doc_id": "docB", "branch_doc_in_selected_support": False},
    ]
