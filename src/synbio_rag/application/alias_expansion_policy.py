from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


class AliasExpansionPolicy:
    def expand(self, query: str, config: Any) -> str:
        return expand_alias_query(query, config)


_ALIAS_MAP_CACHE: dict[str, Any] | None = None


def _load_alias_map(config: Any) -> dict[str, Any] | None:
    """Load alias map from YAML file, cached at module level."""
    global _ALIAS_MAP_CACHE
    if _ALIAS_MAP_CACHE is not None:
        return _ALIAS_MAP_CACHE
    map_path = getattr(config, "alias_expansion_map_path", "") or ""
    if not map_path:
        map_path = str(
            Path(__file__).resolve().parents[3]
            / "src/synbio_rag/resources/retrieval_aliases_v1.yaml"
        )
    try:
        if _HAS_YAML:
            with open(map_path, encoding="utf-8") as f:
                _ALIAS_MAP_CACHE = yaml.safe_load(f)
        else:
            import json

            with open(map_path.replace(".yaml", ".json"), encoding="utf-8") as f:
                _ALIAS_MAP_CACHE = json.load(f)
    except Exception:
        _ALIAS_MAP_CACHE = {}
    return _ALIAS_MAP_CACHE


def expand_alias_query(query: str, config: Any) -> str:
    """Expand query with controlled BM25-only alias terms.

    Trigger-based: only adds expansions when trigger terms appear in query.
    BM25-only: does not modify dense query or LLM prompt.
    Feature-flagged: only runs when alias_expansion_enabled=True.
    """
    if not getattr(config, "alias_expansion_enabled", False):
        return query

    alias_map = _load_alias_map(config)
    if not alias_map:
        return query

    aliases = alias_map.get("aliases", {})
    if not aliases:
        return query

    allowed_risks = set(getattr(config, "alias_expansion_risk_levels", ["low"]))
    max_entities = getattr(config, "alias_expansion_max_entities_per_query", 3)
    max_per_entity = getattr(config, "alias_expansion_max_expansions_per_entity", 3)
    max_total = getattr(config, "alias_expansion_max_total_terms", 8)

    query_lower = query.lower()
    query_normalized = query_lower.replace("\u2019", "'").replace("\u2018", "'").replace("\u2032", "'")

    expansion_terms: list[str] = []
    triggered_count = 0

    for canonical_id, entry in aliases.items():
        if triggered_count >= max_entities:
            break
        risk = entry.get("risk", "medium")
        if risk not in allowed_risks:
            continue

        triggered = False
        for lang, terms in (entry.get("triggers") or {}).items():
            for term in terms:
                if term.lower() in query_lower or term.lower() in query_normalized:
                    triggered = True
                    break
            if triggered:
                break
        if not triggered:
            continue

        expansions = entry.get("expansions", [])[:max_per_entity]
        for exp in expansions:
            if exp.lower() in query_lower or exp.lower() in query_normalized:
                continue
            expansion_terms.append(exp)

        triggered_count += 1
        if len(expansion_terms) >= max_total:
            break

    expansion_terms = expansion_terms[:max_total]
    if not expansion_terms:
        return query

    return f"{query} {' '.join(expansion_terms)}"


_expand_alias_query = expand_alias_query
