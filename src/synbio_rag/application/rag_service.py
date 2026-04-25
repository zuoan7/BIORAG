from __future__ import annotations

import uuid

from ..domain.config import Settings
from ..domain.schemas import ConversationTurn, QueryFilters, RAGResponse
from ..infrastructure.persistence.audit import AuditLogger, to_serializable
from ..infrastructure.persistence.session_store import SessionStore
from .pipeline import SynBioRAGPipeline


class RAGApplicationService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipeline = SynBioRAGPipeline(settings)
        self.sessions = SessionStore(settings.audit)
        self.audit = AuditLogger(settings.audit)

    def ask(
        self,
        question: str,
        session_id: str | None = None,
        filters: QueryFilters | None = None,
        include_debug: bool = False,
    ) -> RAGResponse:
        session_id = session_id or str(uuid.uuid4())
        active_filters = filters or QueryFilters()
        history = self.sessions.get_turns(session_id)
        response = self.pipeline.answer(
            question=question,
            session_id=session_id,
            history=history,
            filters=active_filters,
        )
        self.sessions.append_turn(session_id, ConversationTurn(role="user", content=question))
        self.sessions.append_turn(session_id, ConversationTurn(role="assistant", content=response.answer))
        if not include_debug:
            response.debug = {}
        self.audit.log(
            "rag_ask",
            {
                "session_id": session_id,
                "question": question,
                "filters": to_serializable(active_filters),
                "response": to_serializable(response),
            },
        )
        return response

    def get_session(self, session_id: str) -> list[ConversationTurn]:
        return self.sessions.get_turns(session_id)
