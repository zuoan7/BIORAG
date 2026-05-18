from __future__ import annotations

import re

from .models import EvidenceCandidate, SupportItem


_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][a-z]{2,}(?:[A-Z][a-z]+)*\d*[A-Z]?)\b"
    r"|(?:CRISPR|Cas\d+|FadL|ABC|MFS|RND|MATE|SMR|TRAP|UGGT|HAC1"
    r"|pfkA|zwf|Fam20[A-C]?|Neu5Ac|GDP|UDP|CMP|ATP|NADH|NADPH"
    r"|SpMAE|TsaM|TsaT)",
    re.IGNORECASE,
)
_CJK_3GRAM_RE = re.compile(r"[\u4e00-\u9fff]{3}")
_CJK_4GRAM_RE = re.compile(r"[\u4e00-\u9fff]{4}")
_SECTION_NAMES = {
    "abstract", "introduction", "results", "discussion", "conclusion",
    "conclusions", "methods", "full text", "title", "background",
    "results and discussion", "materials and methods",
}


def _extract_question_entities(question: str) -> set[str]:
    """Extract key entities from question for entity-match fallback.

    Uses English domain terms, longer CJK spans (3-4 gram),
    and candidate title/section field matching to bridge Chinese-English terminology gaps.
    """
    entities: set[str] = set()
    for m in _ENTITY_RE.finditer(question):
        token = m.group()
        if token.lower() not in _SECTION_NAMES and len(token) >= 3:
            entities.add(token.lower())
    # 4-gram CJK (priority — more specific)
    for m in _CJK_4GRAM_RE.finditer(question):
        entities.add(m.group().lower())
    # 3-gram CJK (fallback)
    for m in _CJK_3GRAM_RE.finditer(question):
        entities.add(m.group().lower())
    return entities


def _extract_matched_entities(question: str,
                              candidates: list[EvidenceCandidate]) -> list[str]:
    """Return which question entities matched in candidates."""
    q_entities = _extract_question_entities(question)
    all_text = " ".join(c.text or "" for c in candidates).lower()
    return sorted(e for e in q_entities if e in all_text)


def _build_limited_support_pack(
    question: str,
    candidates: list[EvidenceCandidate],
    max_chunks: int = 3,
) -> list[SupportItem]:
    """When support_pack is empty but candidates have entity hits or minimal rerank scores,
    build a conservative limited_support_pack from top entity-matching or top-scored chunks."""
    q_entities = _extract_question_entities(question)

    # Score each candidate by entity match count (check text + title)
    entity_scored: list[tuple[int, EvidenceCandidate]] = []
    for candidate in candidates:
        text = (candidate.text or "").lower()
        title = (candidate.title or "").lower()
        haystack = text + " " + title
        hits = sum(1 for e in q_entities if e in haystack)
        if hits > 0:
            entity_scored.append((hits, candidate))

    # Fallback: if no entity matches, use top-ranked candidates by rerank score.
    # Candidates were already filtered by retrieval+rerank pipeline, so even
    # low absolute scores indicate relative relevance.
    if not entity_scored:
        ranked = sorted(
            candidates,
            key=lambda c: c.rerank_score or c.fusion_score or 0.0,
            reverse=True,
        )
        # Only use candidates with valid scores (not None/0 for both score types)
        ranked = [c for c in ranked if (c.rerank_score or c.fusion_score or 0.0) != 0.0]
        if not ranked:
            return []
        limited: list[SupportItem] = []
        for candidate in ranked[:max_chunks]:
            score = candidate.rerank_score or candidate.fusion_score or 0.0
            limited.append(SupportItem(
                evidence_id=candidate.evidence_id,
                candidate=candidate,
                support_score=score,
                reasons=["limited_support_pack_fallback", f"rank_fallback_score:{score:.3f}"],
            ))
        return limited

    # Take top-N by entity hits
    entity_scored.sort(key=lambda x: x[0], reverse=True)
    limited: list[SupportItem] = []
    for hits, candidate in entity_scored[:max_chunks]:
        support_score = candidate.rerank_score or candidate.fusion_score or 0.0
        if support_score == 0.0:
            support_score = 0.3  # minimal score for limited pack
        limited.append(SupportItem(
            evidence_id=candidate.evidence_id,
            candidate=candidate,
            support_score=support_score,
            reasons=["limited_support_pack_fallback", f"entity_hits:{hits}"],
        ))

    return limited
