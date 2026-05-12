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

        negative_intent = False
        abstain_clause = False
        intent_before = ""

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

        # Phase 21A-9I: negative/no-answer guard — detect explicit abstain clause
        # Must run AFTER comparison/summary detection to avoid intercepting
        # "compare whether X has Y" or "summarize which ones contain Z".
        # Only triggers when the query BOTH:
        # 1. Asks about existence/evidence ("是否有/是否包含/does the/are there")
        # 2. Has an explicit abstain-if-not clause ("如果没有/if not, explicitly state")
        cn_existence = any(t in q for t in ["是否有", "是否包含", "是否存在", "是否有关于"])
        cn_abstain = any(t in q for t in [
            "如果没有", "请明确说明", "如果不存在", "若无相关证据",
            "如果没有，请", "如果文献中没有", "如果没有找到",
            "若没有", "请说明没有", "请不要推断", "请勿推断",
        ])
        en_existence = any(t in q for t in [
            "does the", "is there", "are there",
            "whether there is", "whether the",
        ])
        en_abstain = any(t in q for t in [
            "if not, explicitly state", "if no evidence",
            "if absent, state", "do not infer if",
            "explicitly state that it does not",
            "state that it does not",
            "state that evidence is insufficient",
        ])

        if (cn_existence and cn_abstain) or (en_existence and en_abstain):
            negative_intent = True
            abstain_clause = True
            intent_before = intent.value
            intent = QueryIntent.NEGATIVE
            notes = "negative/no-answer query with explicit abstain clause"

        return QueryAnalysis(
            intent=intent,
            requires_external_tools=requires_external_tools,
            search_limit=search_limit,
            rerank_top_k=rerank_top_k,
            notes=notes,
            negative_intent_detected=negative_intent,
            abstain_clause_detected=abstain_clause,
            intent_before_negative_guard=intent_before,
        )
