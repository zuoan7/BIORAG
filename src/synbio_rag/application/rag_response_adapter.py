from __future__ import annotations

from ..domain.schemas import RAGPipelineResponse, RAGResponse


def to_legacy_rag_response(response: RAGPipelineResponse) -> RAGResponse:
    return RAGResponse(
        answer=response.answer,
        confidence=response.confidence,
        route=response.route,
        citations=list(response.citations),
        used_external_tool=False,
        tool_name=None,
        tool_result=None,
        session_id=response.session_id,
        external_references=[],
        debug=dict(response.debug),
    )
