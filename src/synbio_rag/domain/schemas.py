from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryIntent(str, Enum):
    FACTOID = "factoid"
    SUMMARY = "summary"
    COMPARISON = "comparison"
    EXPERIMENT = "experiment"
    UNKNOWN = "unknown"


@dataclass
class QueryAnalysis:
    intent: QueryIntent
    requires_external_tools: bool
    search_limit: int
    rerank_top_k: int
    notes: str = ""


@dataclass
class QueryFilters:
    tenant_id: str = "default"
    doc_ids: list[str] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    min_score: float | None = None


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    source_file: str
    title: str
    section: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    fusion_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    chunk_id: str
    doc_id: str
    title: str
    source_file: str
    section: str
    page_start: int | None
    page_end: int | None
    score: float
    quote: str


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class ToolReference:
    source: str
    title: str
    link: str
    snippet: str


@dataclass
class RAGResponse:
    answer: str
    confidence: float
    route: QueryIntent
    citations: list[Citation]
    used_external_tool: bool
    tool_name: str | None
    tool_result: str | None
    session_id: str | None = None
    external_references: list[ToolReference] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
