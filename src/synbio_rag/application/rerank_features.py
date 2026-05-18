from __future__ import annotations

import re

from ..domain.config import RetrievalConfig
from ..domain.schemas import RetrievedChunk
from .rerank_common import _rerank_text


_STRATEGY_GROUPS: dict[str, tuple[str, ...]] = {
    "salvage": ("salvage", "gdp-l-fucose", "gdp fucose", "补料", "补救"),
    "wcfb": ("wcfb", "fucosyltransferase", "末端转移", "岩藻糖基转移酶"),
    "chromosomal integration": (
        "chromosomal integration",
        "chromosomally integrated",
        "chromosomally integration",
        "染色体整合",
        "integrated expression cassette",
    ),
}
_ROUTE_INTENT_MARKERS = ("工程化", "合成路径", "关键前体", "催化步骤", "biosynthesis", "pathway", "precursor")
_ROUTE_POSITIVE_TERMS = (
    "biosynthesis",
    "production",
    "engineered",
    "metabolic engineering",
    "synthesis",
    "pathway",
    "cmp-neu5ac",
    "gdp-l-fucose",
    "gdp-fucose",
    "sialyltransferase",
    "fucosyltransferase",
)
_ROUTE_NEGATIVE_TERMS = (
    "protective effect",
    "covid",
    "sars",
    "necrotizing",
    "inflammation",
    "treatment",
    "review",
    "progress",
    "application",
    "health effects",
)
_NUMERIC_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:%|g/L|mg/L|mg|g|mM|uM|µM|h|hr|hours?|fold|times?|x\b)",
    flags=re.IGNORECASE,
)
_RESULT_TERMS = (
    "yield",
    "production",
    "produced",
    "titer",
    "titre",
    "result",
    "increase",
    "increased",
    "decrease",
    "decreased",
    "improved",
    "enhanced",
    "产量",
    "产率",
    "滴度",
    "提高",
    "降低",
    "增加",
    "减少",
    "结果",
)
_DEFINITION_TERMS = (
    "defined as",
    "is defined",
    "acts as",
    "act as",
    "functions as",
    "function as",
    "refers to",
    "known as",
    "termed",
    "指",
    "定义为",
)
_TABLE_QUERY_HINTS = ("table", "primer", "sequence", "strain", "vmax", "km", "relative peak area", "glycan", "parameter")
_FIGURE_QUERY_HINTS = ("figure", "fig.", "fig ", "图")


def _strategy_bonus(question: str, chunk: RetrievedChunk, config: RetrievalConfig) -> float:
    lowered_question = question.lower()
    haystack = "\n".join(
        part.lower()
        for part in (chunk.title, chunk.section, chunk.text[:800])
        if part
    )
    bonus = 0.0
    for aliases in _STRATEGY_GROUPS.values():
        if not any(alias.lower() in lowered_question for alias in aliases):
            continue
        if any(alias.lower() in haystack for alias in aliases):
            bonus += config.rerank_strategy_bonus
    return bonus


def _route_bonus(question: str, chunk: RetrievedChunk, config: RetrievalConfig) -> float:
    lowered_question = question.lower()
    if not any(marker.lower() in lowered_question for marker in _ROUTE_INTENT_MARKERS):
        return 0.0
    haystack = "\n".join(
        part.lower()
        for part in (chunk.title, chunk.section, chunk.text[:1200])
        if part
    )
    bonus = 0.0
    positive_hits = sum(1 for term in _ROUTE_POSITIVE_TERMS if term in haystack)
    negative_hits = sum(1 for term in _ROUTE_NEGATIVE_TERMS if term in haystack)
    if positive_hits:
        bonus += min(positive_hits, 3) * (config.rerank_strategy_bonus * 0.4)
    if negative_hits:
        bonus -= min(negative_hits, 2) * (config.rerank_strategy_bonus * 0.5)
    if ("前体" in question or "precursor" in lowered_question) and any(
        term in haystack for term in ("gdp-l-fucose", "gdp-fucose", "cmp-neu5ac")
    ):
        bonus += config.rerank_strategy_bonus * 5.0
    if ("催化步骤" in question or "catalytic" in lowered_question or "末端" in question) and any(
        term in haystack for term in ("fucosyltransferase", "sialyltransferase", "alpha-1,2-ft", "alpha2,6")
    ):
        bonus += config.rerank_strategy_bonus
    if any(term in lowered_question for term in ("2′-fl", "2'-fl")) and any(
        term in haystack for term in ("salvage pathway", "gdp-l-fucose")
    ):
        bonus += config.rerank_strategy_bonus * 0.8
    return bonus


def _evidence_aware_bonus(chunk: RetrievedChunk, config: RetrievalConfig) -> float:
    text = _rerank_text(chunk)
    lowered = f" {text.lower()} "
    numeric_feature = 1.0 if _NUMERIC_RE.search(text) or re.search(r"\d+", text) else 0.0
    result_feature = 1.0 if any(term in lowered for term in _RESULT_TERMS) else 0.0
    definition_feature = 1.0 if any(term in lowered for term in _DEFINITION_TERMS) else 0.0
    section_bonus = _section_bonus(chunk.section, config)
    bonus = (
        config.evidence_numeric_bonus * numeric_feature
        + config.evidence_result_bonus * result_feature
        + config.evidence_definition_bonus * definition_feature
        + section_bonus
    )
    chunk.metadata["evidence_features"] = {
        "numeric": bool(numeric_feature),
        "result": bool(result_feature),
        "definition": bool(definition_feature),
        "section_bonus": round(section_bonus, 4),
        "bonus": round(bonus, 4),
    }
    return bonus


def _structure_marker_bonus(question: str, chunk: RetrievedChunk, config: RetrievalConfig) -> float:
    lowered_question = question.lower()
    text = _rerank_text(chunk).lower()
    bonus = 0.0
    if any(hint in lowered_question for hint in _TABLE_QUERY_HINTS):
        if "[table text]" in text:
            bonus += config.table_text_boost
        if "[table caption]" in text:
            bonus += config.table_caption_boost
    if any(hint in lowered_question for hint in _FIGURE_QUERY_HINTS) and "[figure caption]" in text:
        bonus += config.figure_caption_boost
    return bonus


def _section_bonus(section: str, config: RetrievalConfig) -> float:
    normalized = (section or "").strip().lower()
    if "result" in normalized:
        return config.section_results_bonus
    if "discussion" in normalized:
        return config.section_discussion_bonus
    if normalized == "abstract":
        return config.section_abstract_bonus
    if "introduction" in normalized:
        return config.section_introduction_penalty
    return 0.0
