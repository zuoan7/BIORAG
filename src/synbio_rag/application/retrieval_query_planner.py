from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..domain.schemas import QueryAnalysis, QueryIntent
from .query_semantics import (
    _contains_cjk,
    _expand_query_aliases,
    _expand_route_pathway_terms,
    _extract_comparison_subqueries,
)


@dataclass(frozen=True)
class QueryVariant:
    query: str
    weight: float
    kind: str

    def __getitem__(self, key: str) -> object:
        if key not in {"query", "weight", "kind"}:
            raise KeyError(key)
        return getattr(self, key)

    def as_dict(self) -> dict[str, object]:
        return {"query": self.query, "weight": self.weight, "kind": self.kind}


class RetrievalQueryPlanner:
    def build(
        self,
        question: str,
        analysis: QueryAnalysis | None,
        config: Any,
        decomposition_query: str | None = None,
    ) -> list[QueryVariant]:
        return build_query_plan(question, analysis, config, decomposition_query)

    def contains_cjk(self, text: str) -> bool:
        return _contains_cjk(text)


def build_query_plan(
    question: str,
    analysis: QueryAnalysis | None,
    config: Any,
    decomposition_query: str | None = None,
) -> list[QueryVariant]:
    """Build query plan for hybrid retrieval.

    question: the main retrieval query (may be rewritten EN).
    decomposition_query: optional original query for comparison subquery extraction.
        When QUERY_REWRITE_MODE=enabled, retrieval uses EN query but decomposition
        should use the original CN query which preserves comparison structure.
    """
    main_query = _expand_route_pathway_terms(_expand_query_aliases(question))
    plan = [
        QueryVariant(
            query=main_query,
            weight=config.comparison_query_weight,
            kind="original",
        )
    ]
    if not analysis or analysis.intent != QueryIntent.COMPARISON:
        return [QueryVariant(query=main_query, weight=1.0, kind="original")]

    compare_source = decomposition_query if decomposition_query else question
    for subquery in _extract_comparison_subqueries(compare_source):
        if subquery and subquery != compare_source:
            expanded = _expand_query_aliases(subquery)
            plan.append(
                QueryVariant(
                    query=_expand_route_pathway_terms(expanded),
                    weight=config.comparison_subquery_weight,
                    kind="subquery",
                )
            )
    return plan


_build_query_plan = build_query_plan
