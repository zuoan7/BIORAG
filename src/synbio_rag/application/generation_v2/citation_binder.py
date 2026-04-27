from __future__ import annotations

import re

from ...domain.schemas import Citation
from .models import SupportItem

_EVIDENCE_REF_PATTERN = re.compile(r"\[(E\d+)\]")


class CitationBinder:
    def bind(self, answer: str, support_pack: list[SupportItem]) -> tuple[str, list[Citation], dict]:
        support_by_id = {item.evidence_id: item for item in support_pack}
        ordered_ids: list[str] = []
        invalid_ids: list[str] = []

        def replace(match: re.Match[str]) -> str:
            evidence_id = match.group(1)
            item = support_by_id.get(evidence_id)
            if item is None:
                invalid_ids.append(evidence_id)
                return ""
            if evidence_id not in ordered_ids:
                ordered_ids.append(evidence_id)
            return f"[{ordered_ids.index(evidence_id) + 1}]"

        final_answer = _EVIDENCE_REF_PATTERN.sub(replace, answer)
        citations = [self._to_citation(support_by_id[evidence_id]) for evidence_id in ordered_ids]
        debug = {
            "ordered_evidence_ids": ordered_ids,
            "invalid_evidence_ids": invalid_ids,
        }
        return final_answer, citations, debug

    def _to_citation(self, item: SupportItem) -> Citation:
        candidate = item.candidate
        return Citation(
            chunk_id=candidate.chunk_id,
            doc_id=candidate.doc_id,
            title=candidate.title,
            source_file=candidate.source_file,
            section=candidate.section,
            page_start=candidate.page_start,
            page_end=candidate.page_end,
            score=item.support_score,
            quote=_compress_quote(candidate.text),
        )


def _compress_quote(text: str) -> str:
    quote = " ".join((text or "").split())
    if len(quote) <= 1200:
        return quote
    return quote[:1197].rstrip() + "..."
