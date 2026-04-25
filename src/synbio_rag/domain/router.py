from __future__ import annotations

from .config import RetrievalConfig
from .schemas import QueryAnalysis, QueryIntent


class QueryRouter:
    def __init__(self, retrieval_config: RetrievalConfig):
        self.retrieval_config = retrieval_config

    def analyze(self, question: str) -> QueryAnalysis:
        q = question.lower()
        intent = QueryIntent.UNKNOWN
        requires_external_tools = False
        search_limit = self.retrieval_config.search_limit
        rerank_top_k = self.retrieval_config.rerank_top_k
        notes = ""

        if any(token in q for token in ["compare", "difference", "versus", "相比", "比较", "区别", "对比", "差异"]):
            intent = QueryIntent.COMPARISON
            search_limit += 8
            rerank_top_k += 2
            notes = "comparison query expands recall"
        elif any(token in q for token in ["summary", "综述", "总结", "概述", "概括", "梳理", "归纳"]):
            intent = QueryIntent.SUMMARY
            search_limit += 5
            notes = "summary query prefers broader recall"
        elif any(token in q for token in ["protocol", "步骤", "实验", "construct", "design", "流程", "方案设计"]):
            intent = QueryIntent.EXPERIMENT
            search_limit += 5
            requires_external_tools = True
            notes = "experimental query may need external protocol tools"
        elif any(
            token in q
            for token in [
                "what",
                "how",
                "why",
                "是否",
                "能否",
                "是什么",
                "有没有",
                "哪些",
                "哪种",
                "哪两个",
                "哪几种",
                "为什么",
                "原因",
                "作用",
                "机制",
                "如何",
                "请只基于证据回答",
                "根据文库",
            ]
        ):
            intent = QueryIntent.FACTOID
            notes = "factoid query uses standard recall"
        elif "?" in question or "？" in question:
            intent = QueryIntent.FACTOID
            notes = "fallback question mark heuristic"

        return QueryAnalysis(
            intent=intent,
            requires_external_tools=requires_external_tools,
            search_limit=search_limit,
            rerank_top_k=rerank_top_k,
            notes=notes,
        )
