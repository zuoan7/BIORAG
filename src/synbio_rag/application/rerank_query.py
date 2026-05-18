from __future__ import annotations

from ..domain.schemas import QueryAnalysis, QueryIntent
from .query_semantics import _extract_comparison_subqueries, _expand_query_aliases


def _build_rerank_queries(question: str, analysis: QueryAnalysis | None) -> list[str]:
    queries = [_expand_query_aliases(question)]
    if not analysis or analysis.intent != QueryIntent.COMPARISON:
        return queries
    for subquery in _extract_comparison_subqueries(question):
        expanded = _expand_query_aliases(subquery)
        if expanded and expanded not in queries:
            queries.append(expanded)
    return queries
