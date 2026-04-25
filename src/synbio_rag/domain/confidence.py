from __future__ import annotations

import math

from .config import ConfidenceConfig
from .schemas import RetrievedChunk


class ConfidenceScorer:
    def __init__(self, config: ConfidenceConfig):
        self.config = config

    def score(self, chunks: list[RetrievedChunk]) -> float:
        if not chunks:
            return 0.0
        top_scores = [_normalize_score(max(chunk.rerank_score, chunk.vector_score, chunk.fusion_score)) for chunk in chunks[:3]]
        avg = sum(top_scores) / len(top_scores)
        diversity_bonus = min(len({chunk.doc_id for chunk in chunks[:3]}) * 0.05, 0.15)
        return min(avg + diversity_bonus, 0.99)

    def needs_external_tool(self, confidence: float) -> bool:
        return confidence < self.config.answer_threshold


def _normalize_score(score: float) -> float:
    if 0.0 <= score <= 1.0:
        return score
    return 1.0 / (1.0 + math.exp(-score))
