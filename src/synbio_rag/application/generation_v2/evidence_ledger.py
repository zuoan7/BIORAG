from __future__ import annotations

import re

from ...domain.schemas import QueryAnalysis, RetrievedChunk
from .models import EvidenceCandidate

_RESULT_PATTERN = re.compile(
    r"\b(result|results|showed|demonstrated|increased|decreased|yield|titer|production)\b|结果|显示|提高|降低|产量|滴度",
    re.IGNORECASE,
)
_DIGIT_PATTERN = re.compile(r"\d")


class EvidenceLedgerBuilder:
    def build(
        self,
        question: str,
        analysis: QueryAnalysis,
        seed_chunks: list[RetrievedChunk],
    ) -> list[EvidenceCandidate]:
        del question, analysis
        candidates: list[EvidenceCandidate] = []
        for index, chunk in enumerate(seed_chunks, start=1):
            text = chunk.text or ""
            metadata = dict(chunk.metadata or {})
            lower_text = text.lower()
            lower_section = (chunk.section or "").strip().lower()
            feature_flags = {
                "has_table_text": "[table text]" in lower_text or "table_text" in lower_text or "table_text" in metadata,
                "has_table_caption": "[table caption]" in lower_text or "table_caption" in lower_text or "table_caption" in metadata,
                "has_figure_caption": "[figure caption]" in lower_text or "fig." in lower_text or "figure_caption" in lower_text or "figure_caption" in metadata,
                "has_numeric": bool(_DIGIT_PATTERN.search(text)),
                "has_result_terms": bool(_RESULT_PATTERN.search(text)),
                "section_type": lower_section,
                "text_length": len(text),
            }
            reasons = [f"seed_chunk", f"section:{lower_section or 'unknown'}"]
            for feature_name, enabled in feature_flags.items():
                if feature_name in {"section_type", "text_length"} or not enabled:
                    continue
                reasons.append(feature_name)
            candidates.append(
                EvidenceCandidate(
                    evidence_id=f"E{index}",
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_file=chunk.source_file,
                    title=chunk.title,
                    section=chunk.section,
                    text=chunk.text,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    vector_score=chunk.vector_score,
                    bm25_score=chunk.bm25_score,
                    rerank_score=chunk.rerank_score,
                    fusion_score=chunk.fusion_score,
                    metadata=metadata,
                    features=feature_flags,
                    reasons=reasons,
                )
            )
        return candidates
