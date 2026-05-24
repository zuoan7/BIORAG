from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synbio_rag.application.generation_v2.models import EvidenceCandidate  # noqa: E402
from synbio_rag.application.generation_v2.support_selector import SupportPackSelector  # noqa: E402
from synbio_rag.domain.config import GenerationConfig  # noqa: E402
from synbio_rag.domain.schemas import QueryAnalysis, QueryIntent  # noqa: E402


def test_factoid_appends_matched_child_candidate_to_plus_one_capacity() -> None:
    support = _select(
        QueryIntent.FACTOID,
        [
            _candidate("e1", 1.0, "doc-a", 1),
            _candidate("e2", 0.9, "doc-b", 2),
            _candidate("e3", 0.8, "doc-c", 3),
            _candidate("e4", 0.7, "doc-a", 8, matched_child=True),
        ],
    )

    assert [item.evidence_id for item in support] == ["e1", "e2", "e3", "e4"]
    assert "matched_child_support_retention_append" in support[-1].reasons


def test_factoid_swaps_low_unprotected_item_when_plus_one_capacity_is_full() -> None:
    support = _select(
        QueryIntent.FACTOID,
        [
            _candidate("e1", 1.0, "doc-a", 1),
            _candidate("e2", 0.9, "doc-b", 2),
            _candidate("e3", 0.5, "doc-c", 9),
            _candidate("e4", 0.45, "doc-d", 10),
            _candidate("e5", 0.44, "doc-a", 8, matched_child=True),
        ],
    )

    selected_ids = [item.evidence_id for item in support]
    assert selected_ids == ["e1", "e2", "e3", "e5"]
    assert "e4" not in selected_ids
    assert "matched_child_support_retention_swap" in support[-1].reasons


def test_factoid_does_not_replace_protected_top_two_seeds() -> None:
    support = _select(
        QueryIntent.FACTOID,
        [
            _candidate("e1", 0.30, "doc-a", 1),
            _candidate("e2", 0.29, "doc-b", 2),
            _candidate("e3", 0.28, "doc-c", 3, matched_child=True),
            _candidate("e4", 0.25, "doc-d", 4, matched_child=True),
            _candidate("e5", 0.24, "doc-a", 8, matched_child=True),
        ],
    )

    selected_ids = [item.evidence_id for item in support]
    assert selected_ids == ["e1", "e2", "e3", "e4"]
    assert "e5" not in selected_ids


def test_factoid_does_not_retain_below_min_score_matched_child_candidate() -> None:
    support = _select(
        QueryIntent.FACTOID,
        [
            _candidate("e1", 0.9, "doc-a", 1),
            _candidate("e2", 0.8, "doc-b", 2),
            _candidate("e3", 0.7, "doc-c", 3),
            _candidate("e4", 0.49, "doc-a", 8, matched_child=True),
        ],
        config=GenerationConfig(v2_min_support_score=0.5),
    )

    assert [item.evidence_id for item in support] == ["e1", "e2", "e3"]


def test_negative_route_still_returns_empty_support_pack() -> None:
    selector = SupportPackSelector()
    support = selector.select(
        "zzquery",
        _analysis(QueryIntent.NEGATIVE),
        [_candidate("e1", 1.0, "doc-a", 1, matched_child=True)],
        GenerationConfig(),
    )

    assert support == []
    assert selector.last_selection_debug["negative_guard"] is True


def test_summary_and_comparison_routes_do_not_apply_matched_child_retention() -> None:
    summary_support = _select(
        QueryIntent.SUMMARY,
        [
            _candidate("s1", 1.0, "doc-a", 1, section="Abstract"),
            _candidate("s2", 0.9, "doc-b", 2, section="Abstract"),
            _candidate("s3", 0.8, "doc-c", 3, section="Abstract"),
            _candidate("s4", 0.7, "doc-a", 8, section="Results", matched_child=True),
        ],
        question="summarize zzquery",
    )
    comparison_support = _select(
        QueryIntent.COMPARISON,
        [
            _candidate("c1", 1.0, "doc-a", 1),
            _candidate("c2", 0.9, "doc-b", 2),
            _candidate("c3", 0.8, "doc-c", 3),
            _candidate("c4", 0.7, "doc-a", 8, matched_child=True),
        ],
        config=GenerationConfig(v2_max_support_comparison=3),
        question="comparison overview",
    )

    assert [item.evidence_id for item in summary_support] == ["s1", "s2", "s3"]
    assert [item.evidence_id for item in comparison_support] == ["c1", "c2", "c3"]
    assert not _has_retention_reason(summary_support)
    assert not _has_retention_reason(comparison_support)


def _select(
    intent: QueryIntent,
    candidates: list[EvidenceCandidate],
    *,
    config: GenerationConfig | None = None,
    question: str = "zzquery",
) -> list:
    return SupportPackSelector().select(
        question,
        _analysis(intent),
        candidates,
        config or GenerationConfig(),
    )


def _analysis(intent: QueryIntent) -> QueryAnalysis:
    return QueryAnalysis(
        intent=intent,
        requires_external_tools=False,
        search_limit=10,
        rerank_top_k=10,
    )


def _candidate(
    evidence_id: str,
    score: float,
    doc_id: str,
    rerank_rank: int,
    *,
    matched_child: bool = False,
    section: str = "Results",
) -> EvidenceCandidate:
    metadata = {"rerank_rank": rerank_rank, "matched_child_chunk_ids": []}
    if matched_child:
        metadata["matched_child_chunk_ids"] = [f"{evidence_id}-child"]
    return EvidenceCandidate(
        evidence_id=evidence_id,
        chunk_id=f"{evidence_id}-chunk",
        doc_id=doc_id,
        source_file=f"{doc_id}.pdf",
        title=f"Title {doc_id}",
        section=section,
        text=f"support passage for {evidence_id} with enough text",
        page_start=1,
        page_end=1,
        vector_score=0.0,
        bm25_score=0.0,
        rerank_score=score,
        fusion_score=0.0,
        metadata=metadata,
        features={"text_length": 48},
        reasons=[],
    )


def _has_retention_reason(support: list) -> bool:
    return any(
        reason.startswith("matched_child_support_retention")
        for item in support
        for reason in item.reasons
    )
