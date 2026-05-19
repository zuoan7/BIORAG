from __future__ import annotations

from types import SimpleNamespace

from src.synbio_rag.application.rag_service import RAGApplicationService
from src.synbio_rag.application.rag_response_adapter import to_legacy_rag_response
from src.synbio_rag.domain.schemas import (
    Citation,
    ConversationTurn,
    QueryIntent,
    RAGPipelineResponse,
    RAGResponse,
)


def test_adapter_adds_legacy_external_tool_fields() -> None:
    citation = Citation(
        chunk_id="c1",
        doc_id="d1",
        title="title",
        source_file="paper.pdf",
        section="Results",
        page_start=1,
        page_end=2,
        score=0.8,
        quote="evidence",
    )
    internal = RAGPipelineResponse(
        answer="ok",
        confidence=0.7,
        route=QueryIntent.FACTOID,
        citations=[citation],
        session_id="session-1",
        debug={"generation_v2": {"mode": "v2"}},
    )

    legacy = to_legacy_rag_response(internal)

    assert isinstance(legacy, RAGResponse)
    assert legacy.answer == "ok"
    assert legacy.confidence == 0.7
    assert legacy.route == QueryIntent.FACTOID
    assert legacy.citations == [citation]
    assert legacy.session_id == "session-1"
    assert legacy.debug == {"generation_v2": {"mode": "v2"}}
    assert legacy.used_external_tool is False
    assert legacy.tool_name is None
    assert legacy.tool_result is None
    assert legacy.external_references == []


def test_application_service_returns_legacy_response_dto() -> None:
    logged = {}
    turns: list[ConversationTurn] = []
    pipeline_response = RAGPipelineResponse(
        answer="ok",
        confidence=0.7,
        route=QueryIntent.FACTOID,
        citations=[],
        session_id="session-1",
        debug={"trace": True},
    )
    service = RAGApplicationService.__new__(RAGApplicationService)
    service.pipeline = SimpleNamespace(answer=lambda **kwargs: pipeline_response)
    service.sessions = SimpleNamespace(
        get_turns=lambda session_id: [],
        append_turn=lambda session_id, turn: turns.append(turn),
    )
    service.audit = SimpleNamespace(log=lambda event, payload: logged.update(payload))

    response = service.ask("question?", session_id="session-1", include_debug=False)

    assert isinstance(response, RAGResponse)
    assert response.used_external_tool is False
    assert response.external_references == []
    assert response.debug == {}
    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert logged["response"]["used_external_tool"] is False
