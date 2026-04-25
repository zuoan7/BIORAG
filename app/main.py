from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag import RAGApplicationService, Settings
from src.synbio_rag.domain.schemas import QueryFilters


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache
def get_service() -> RAGApplicationService:
    return RAGApplicationService(get_settings())


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.2.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, description="用户问题")
    session_id: str | None = Field(default=None, description="多轮会话 ID")
    tenant_id: str = Field(default="default", description="租户标识")
    doc_ids: list[str] = Field(default_factory=list, description="限定文档 ID")
    sections: list[str] = Field(default_factory=list, description="限定章节")
    source_files: list[str] = Field(default_factory=list, description="限定源文件名")
    min_score: float | None = Field(default=None, description="最低相似度阈值")
    include_debug: bool = Field(default=False, description="是否返回调试信息")


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "collection": settings.retrieval.collection_name,
    }


@app.get("/v1/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    turns = get_service().get_session(session_id)
    return {"session_id": session_id, "turns": [turn.__dict__ for turn in turns]}


@app.post("/v1/ask")
def ask(request: AskRequest) -> dict:
    filters = QueryFilters(
        tenant_id=request.tenant_id,
        doc_ids=request.doc_ids,
        sections=request.sections,
        source_files=request.source_files,
        min_score=request.min_score,
    )
    response = get_service().ask(
        question=request.question,
        session_id=request.session_id,
        filters=filters,
        include_debug=request.include_debug,
    )
    return {
        "session_id": response.session_id,
        "answer": response.answer,
        "confidence": response.confidence,
        "route": response.route.value,
        "citations": [citation.__dict__ for citation in response.citations],
        "used_external_tool": response.used_external_tool,
        "tool_name": response.tool_name,
        "tool_result": response.tool_result,
        "external_references": [item.__dict__ for item in response.external_references],
        "debug": response.debug,
    }
