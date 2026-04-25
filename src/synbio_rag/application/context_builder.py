from __future__ import annotations

from collections import defaultdict

from ..domain.schemas import ConversationTurn, QueryIntent, RetrievedChunk


class ContextBuilder:
    def __init__(self, max_context_chars: int = 32000):
        self.max_context_chars = max_context_chars

    def build(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[ConversationTurn] | None = None,
        intent: QueryIntent = QueryIntent.UNKNOWN,
    ) -> str:
        lines = [
            "你是合成生物学企业知识助手。",
            "回答必须严格基于给定证据，不足时明确说明。",
        ]
        if history:
            lines.append("对话历史:")
            for turn in history[-6:]:
                lines.append(f"{turn.role}: {turn.content}")
            lines.append("")
        lines.extend([f"用户问题: {question}", ""])

        evidence_block = self._build_evidence_block(chunks, intent)
        current_length = sum(len(line) for line in lines)
        if current_length + len(evidence_block) > self.max_context_chars:
            evidence_block = evidence_block[: self.max_context_chars - current_length]
        lines.append(evidence_block)

        return "\n".join(lines)

    def _build_evidence_block(
        self,
        chunks: list[RetrievedChunk],
        intent: QueryIntent,
    ) -> str:
        if not chunks:
            return "=== 证据来源 ===\n（无相关证据）\n=== 证据结束 ==="

        indexed = list(enumerate(chunks, start=1))

        if intent == QueryIntent.COMPARISON:
            return self._format_by_doc(indexed)
        if intent == QueryIntent.SUMMARY:
            return self._format_by_section(indexed)
        return self._format_by_doc(indexed)

    def _format_by_doc(
        self, indexed_chunks: list[tuple[int, RetrievedChunk]]
    ) -> str:
        groups: dict[str, list[tuple[int, RetrievedChunk]]] = defaultdict(list)
        for idx, chunk in indexed_chunks:
            groups[chunk.doc_id].append((idx, chunk))

        lines = ["=== 证据来源 ===", ""]
        for doc_id, items in groups.items():
            title = items[0][1].title or doc_id
            lines.append(f"--- 文档: {title} ---")
            lines.append("")
            for idx, chunk in items:
                page_info = f"Pages: {chunk.page_start}-{chunk.page_end}" if chunk.page_start else ""
                header_parts = [f"Section: {chunk.section or 'N/A'}"]
                if page_info:
                    header_parts.append(page_info)
                lines.append(f"[{idx}] {' | '.join(header_parts)}")
                lines.append(chunk.text)
                lines.append("")
        lines.append("=== 证据结束 ===")
        return "\n".join(lines)

    def _format_by_section(
        self, indexed_chunks: list[tuple[int, RetrievedChunk]]
    ) -> str:
        groups: dict[str, list[tuple[int, RetrievedChunk]]] = defaultdict(list)
        for idx, chunk in indexed_chunks:
            section = chunk.section or "Other"
            groups[section].append((idx, chunk))

        section_order = ["Results", "Discussion", "Methods", "Abstract", "Introduction"]
        ordered_keys = [s for s in section_order if s in groups]
        ordered_keys += [s for s in groups if s not in ordered_keys]

        lines = ["=== 证据来源 ===", ""]
        for section in ordered_keys:
            items = groups[section]
            lines.append(f"--- Section: {section} ---")
            lines.append("")
            for idx, chunk in items:
                title = chunk.title or chunk.doc_id
                page_info = f"Pages: {chunk.page_start}-{chunk.page_end}" if chunk.page_start else ""
                header_parts = [f"来源: {title}"]
                if page_info:
                    header_parts.append(page_info)
                lines.append(f"[{idx}] {' | '.join(header_parts)}")
                lines.append(chunk.text)
                lines.append("")
        lines.append("=== 证据结束 ===")
        return "\n".join(lines)
