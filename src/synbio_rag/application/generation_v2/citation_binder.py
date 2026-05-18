from __future__ import annotations

import re

from ...domain.schemas import Citation
from .models import CitationCandidate, SupportItem

_EVIDENCE_REF_PATTERN = re.compile(r"\[(E\d+)\]")


class CitationBinder:
    def build_citation_candidates(
        self,
        support_pack: list[SupportItem],
        plan_mode: str = "full",
        answer_mode: str = "full",
    ) -> list[CitationCandidate]:
        candidates: list[CitationCandidate] = []
        for item in support_pack:
            c = item.candidate
            metadata = c.metadata or {}
            rank = metadata.get("rerank_rank", 999)
            if not isinstance(rank, (int, float)):
                rank = 999
            rank = int(rank)

            is_protected = isinstance(metadata.get("rerank_rank"), (int, float)) and int(
                metadata.get("rerank_rank", 999)
            ) <= 3

            text_ok = bool(c.text and len(c.text.strip()) >= 10)
            metadata_ok = bool(c.chunk_id and c.doc_id and c.source_file)
            citation_eligible = text_ok and metadata_ok
            table_block_reason = _table_preview_citation_block_reason(
                metadata=metadata,
                source_file=c.source_file,
            )
            if table_block_reason:
                citation_eligible = False

            drop_reason = ""
            if table_block_reason:
                drop_reason = table_block_reason
            elif not metadata_ok:
                drop_reason = "metadata_missing"
            elif not text_ok:
                drop_reason = "text_missing"
            elif not citation_eligible:
                drop_reason = "not_citation_eligible"

            candidate = CitationCandidate(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                source_file=c.source_file,
                title=c.title,
                text=c.text,
                section=c.section,
                page_start=c.page_start,
                page_end=c.page_end,
                answer_mode=answer_mode,
                plan_mode=plan_mode,
                is_from_selected_support=True,
                is_protected_seed=is_protected,
                protected_reason="rerank_top3_seed" if is_protected else "",
                rerank_rank=rank,
                support_priority=item.support_score,
                citation_priority=0.0,
                citation_eligible=citation_eligible,
                evidence_id=item.evidence_id,
                support_score=item.support_score,
                reasons=list(item.reasons),
                drop_reason=drop_reason,
            )
            candidates.append(candidate)

        for candidate in candidates:
            candidate.citation_priority = self._compute_citation_priority(candidate)

        return candidates

    def _compute_citation_priority(self, candidate: CitationCandidate) -> float:
        priority = 0.0

        if candidate.is_protected_seed:
            priority += 3.0

        if candidate.support_priority > 0:
            priority += min(candidate.support_priority * 2.0, 2.0)

        if candidate.rerank_rank > 0 and candidate.rerank_rank <= 10:
            priority += (10 - candidate.rerank_rank) * 0.15

        if candidate.section and "result" in candidate.section.lower():
            priority += 0.5
        if candidate.section and "discussion" in candidate.section.lower():
            priority += 0.4
        if candidate.section and "abstract" in candidate.section.lower():
            priority += 0.3

        return round(priority, 3)

    def bind(
        self,
        answer: str,
        support_pack: list[SupportItem],
        plan_mode: str = "full",
        answer_mode: str = "full",
        citation_candidates: list[CitationCandidate] | None = None,
    ) -> tuple[str, list[Citation], dict]:
        candidates = (
            citation_candidates
            if citation_candidates is not None
            else self.build_citation_candidates(
                support_pack, plan_mode=plan_mode, answer_mode=answer_mode
            )
        )
        candidate_by_eid: dict[str, CitationCandidate] = {}
        for cand in candidates:
            candidate_by_eid[cand.evidence_id] = cand

        ordered_eids: list[str] = []
        invalid_ids: list[str] = []
        blocked_ids: list[str] = []

        def replace(match: re.Match[str]) -> str:
            evidence_id = match.group(1)
            cand = candidate_by_eid.get(evidence_id)
            if cand is None:
                invalid_ids.append(evidence_id)
                return ""
            if not cand.citation_eligible:
                blocked_ids.append(evidence_id)
                if not cand.drop_reason:
                    cand.drop_reason = "not_citation_eligible"
                return ""
            if evidence_id not in ordered_eids:
                ordered_eids.append(evidence_id)
            return f"[{ordered_eids.index(evidence_id) + 1}]"

        final_answer = _EVIDENCE_REF_PATTERN.sub(replace, answer)
        completed_count = 0

        citations = [
            self._to_citation(candidate_by_eid[evidence_id])
            for evidence_id in ordered_eids
            if evidence_id in candidate_by_eid
        ]
        ordered_set = set(ordered_eids)
        cited_chunk_ids = {citation.chunk_id for citation in citations}

        # Compute drop_reasons for uncited candidates
        ordered_set = set(ordered_eids)
        cited_chunk_ids = {citation.chunk_id for citation in citations}
        drop_reasons_by_eid: dict[str, str] = {}
        for candidate in candidates:
            if candidate.chunk_id in cited_chunk_ids:
                continue
            if candidate.drop_reason:
                drop_reasons_by_eid[candidate.evidence_id] = candidate.drop_reason
            elif candidate.evidence_id not in ordered_set:
                drop_reasons_by_eid[candidate.evidence_id] = "citation_marker_not_used"
            else:
                drop_reasons_by_eid[candidate.evidence_id] = "selected_support_not_referenced"

        uncited_eids = [
            candidate.evidence_id
            for candidate in candidates
            if candidate.chunk_id not in cited_chunk_ids
        ]

        debug = {
            "ordered_evidence_ids": ordered_eids,
            "invalid_evidence_ids": invalid_ids,
            "blocked_evidence_ids": blocked_ids,
            "input_evidence_ids": [c.evidence_id for c in candidates],
            "uncited_selected_support_evidence_ids": uncited_eids,
            "drop_reasons_by_evidence_id": drop_reasons_by_eid,
            "citation_candidates": [c.to_dict() for c in candidates],
            "citation_candidate_count": len(candidates),
            "citation_eligible_count": sum(1 for c in candidates if c.citation_eligible),
            "citation_completion_count": completed_count,
            "plan_mode": plan_mode,
        }
        return final_answer, citations, debug

    def _to_citation(self, candidate: CitationCandidate) -> Citation:
        return Citation(
            chunk_id=candidate.chunk_id,
            doc_id=candidate.doc_id,
            title=candidate.title,
            source_file=candidate.source_file,
            section=candidate.section,
            page_start=candidate.page_start,
            page_end=candidate.page_end,
            score=candidate.support_score,
            quote=_compress_quote(candidate.text),
        )


def _compress_quote(text: str) -> str:
    quote = " ".join((text or "").split())
    if len(quote) <= 1200:
        return quote
    return quote[:1197].rstrip() + "..."


def _table_preview_citation_block_reason(
    *,
    metadata: dict,
    source_file: str,
) -> str:
    if metadata.get("object_type") != "table_index_unit":
        return ""
    reasons: list[str] = []
    if metadata.get("table_preview_allow_formal_citation") is not True:
        reasons.append("formal_citation_disabled")
    if metadata.get("production_ready") is not True:
        reasons.append("production_ready_false")
    if metadata.get("index_unit_status") == "preview_only":
        reasons.append("preview_only")
    if metadata.get("value_bboxes_available") is False:
        reasons.append("value_bboxes_unavailable")
    if _looks_like_debug_path(source_file, metadata):
        reasons.append("debug_path_not_formal_source")
    if not reasons:
        return ""
    metadata["table_preview_citation_block_reasons"] = list(reasons)
    return "table_preview_formal_citation_blocked"


def _looks_like_debug_path(source_file: str, metadata: dict) -> bool:
    if not source_file:
        return False
    debug_paths = {
        metadata.get("source_csv_path"),
        metadata.get("source_pdf_crop_path"),
        metadata.get("source_markdown_path"),
    }
    if source_file in debug_paths:
        return True
    return source_file.lower().endswith((".csv", ".png", ".jpg", ".jpeg", ".md"))
