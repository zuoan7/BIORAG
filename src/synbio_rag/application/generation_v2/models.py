from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ...domain.schemas import Citation, RetrievedChunk


@dataclass
class EvidenceCandidate:
    evidence_id: str
    chunk_id: str
    doc_id: str
    source_file: str
    title: str
    section: str
    text: str
    page_start: int | None
    page_end: int | None
    vector_score: float
    bm25_score: float
    rerank_score: float
    fusion_score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SupportItem:
    evidence_id: str
    candidate: EvidenceCandidate
    support_score: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BranchEvidence:
    branch: str
    status: str
    evidence_ids: list[str] = field(default_factory=list)
    primary_evidence_ids: list[str] = field(default_factory=list)
    secondary_evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    missing_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonCoverage:
    parse_ok: bool
    branches: list[str] = field(default_factory=list)
    branch_evidence: list[BranchEvidence] = field(default_factory=list)
    covered_branches: list[str] = field(default_factory=list)
    missing_branches: list[str] = field(default_factory=list)
    allowed_citation_evidence_ids: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerPlan:
    mode: str
    reason: str
    covered_branches: list[str] = field(default_factory=list)
    missing_branches: list[str] = field(default_factory=list)
    allowed_scope: list[str] = field(default_factory=list)
    comparison_coverage: ComparisonCoverage | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationV2Result:
    answer: str
    citations: list[Citation]
    answer_plan: AnswerPlan
    support_pack: list[SupportItem]
    debug: dict[str, Any] = field(default_factory=dict)
    confidence_chunks: list[RetrievedChunk] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [asdict(citation) for citation in self.citations],
            "answer_plan": self.answer_plan.to_dict(),
            "support_pack": [item.to_dict() for item in self.support_pack],
            "debug": self.debug,
            "confidence_chunks": [asdict(chunk) for chunk in self.confidence_chunks],
        }
