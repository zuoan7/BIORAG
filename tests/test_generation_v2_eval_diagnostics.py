from __future__ import annotations

from scripts.evaluation.evaluate_ragas import build_failure_diagnostics, evaluate_retrieval


def _record(*, answer_mode: str, citation_count: int, answer: str) -> dict:
    citations = [
        {
            "chunk_id": "c1",
            "doc_id": "doc1",
            "title": "title",
            "source_file": "doc1.pdf",
            "section": "Results",
            "page_start": None,
            "page_end": None,
            "score": 0.8,
            "quote": "quoted evidence",
        }
        for _ in range(citation_count)
    ]
    return {
        "id": f"sample_{answer_mode}_{citation_count}",
        "question": "test question",
        "reference": "回答应指出证据内容。",
        "response": answer,
        "retrieved_contexts": ["doc1.pdf | Results: quoted evidence"] if citation_count else [],
        "dataset_meta": {
            "expected_doc_ids": ["doc1"],
            "accepted_doc_ids": [],
            "expected_source_files": ["doc1.pdf"],
            "accepted_source_files": [],
            "expected_sections": ["Results"],
            "expected_route": "summary",
            "accepted_routes": [],
            "expected_min_citations": 1,
            "comparison_branches": [],
            "allow_partial_if_doc_coverage": 0,
        },
        "api_response": {
            "route": "summary",
            "citations": citations,
            "debug": {
                "generation_v2": {
                    "answer_mode": answer_mode,
                    "answer_plan": {"mode": answer_mode, "reason": "test_reason"},
                    "support_pack_count": 0 if answer_mode == "refuse" else 2,
                    "validator_debug": {},
                }
            },
        },
    }


def test_v2_diagnostics_use_generation_v2_answer_mode() -> None:
    record = _record(answer_mode="partial", citation_count=1, answer="当前知识库只能支持部分回答。")
    evaluate_retrieval([record], embeddings=None)

    diagnostics = build_failure_diagnostics([record])

    assert diagnostics["samples"]["final_answer_mode_distribution"]["partial"] == 1
    assert "unknown" not in diagnostics["samples"]["final_answer_mode_distribution"]


def test_refuse_with_zero_citation_is_not_counted_as_substantive_zero_citation() -> None:
    record = _record(
        answer_mode="refuse",
        citation_count=0,
        answer="当前知识库证据不足，无法基于已检索证据回答该问题。",
    )
    evaluate_retrieval([record], embeddings=None)

    diagnostics = build_failure_diagnostics([record])

    assert diagnostics["samples"]["refusal_no_citation_ids"] == [record["id"]]
    assert diagnostics["samples"]["zero_citation_substantive_answer_ids"] == []


def test_non_refuse_zero_citation_is_counted_as_substantive_zero_citation() -> None:
    record = _record(
        answer_mode="full",
        citation_count=0,
        answer="根据当前知识库证据：这是一个没有引用的实质回答。",
    )
    evaluate_retrieval([record], embeddings=None)

    diagnostics = build_failure_diagnostics([record])

    assert diagnostics["samples"]["zero_citation_substantive_answer_ids"] == [record["id"]]
    assert diagnostics["samples"]["refusal_no_citation_ids"] == []
