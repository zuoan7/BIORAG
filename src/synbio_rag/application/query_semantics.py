from __future__ import annotations

import re


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_COMPARISON_SPLIT_RE = re.compile(r"\s*(?:,|，|、|以及|及|和|与|vs\.?|versus)\s*")
_TITLE_BOOST_KEYWORDS = (
    "6′-sl",
    "6'-sl",
    "2′-fl",
    "2'-fl",
    "wcfb",
    "salvage",
    "crispr-tmsd",
)
_KEYWORD_ALIASES: dict[str, tuple[str, ...]] = {
    "2′-fl": ("2′-fl", "2'-fl", "2-fucosyllactose", "20-fucosyllactose"),
    "2'-fl": ("2′-fl", "2'-fl", "2-fucosyllactose", "20-fucosyllactose"),
    "6′-sl": ("6′-sl", "6'-sl", "6-sialyllactose", "6′-sialyllactose"),
    "6'-sl": ("6′-sl", "6'-sl", "6-sialyllactose", "6′-sialyllactose"),
    "wcfb": ("wcfb",),
    "salvage": ("salvage", "gdp-l-fucose", "gdp fucose"),
    "crispr-tmsd": ("crispr-tmsd", "tmsd", "strand displacement"),
}
_ROUTE_QUERY_MARKERS = ("工程化", "合成路径", "关键前体", "催化步骤", "biosynthesis", "pathway", "precursor")
_ROUTE_TERM_ALIASES: dict[str, tuple[str, ...]] = {
    "2′-fl": ("gdp-l-fucose", "gdp-fucose", "alpha-1,2-fucosyltransferase", "fuct", "lactose", "biosynthesis"),
    "2'-fl": ("gdp-l-fucose", "gdp-fucose", "alpha-1,2-fucosyltransferase", "fuct", "lactose", "biosynthesis"),
    "6′-sl": ("cmp-neu5ac", "sialyltransferase", "alpha-2,6-sialyltransferase", "lactose", "biosynthesis"),
    "6'-sl": ("cmp-neu5ac", "sialyltransferase", "alpha-2,6-sialyltransferase", "lactose", "biosynthesis"),
}

_ORGANISM_ABBREV_RE = re.compile(
    r"\b([A-Z])\.\s*(coli|subtilis|cerevisiae|pastoris|glutamicum|lactis|amyloliquefaciens|licheniformis|megaterium)\b",
    re.IGNORECASE,
)


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _extract_comparison_subqueries(question: str) -> list[str]:
    normalized = question.strip()
    if not normalized:
        return []
    safe = _mask_organism_abbrevs(normalized)
    prefix_removed = re.sub(
        r"^(?:请|请你|请基于文库|请根据文库)?\s*(?:比较|对比|相比|compare)\s*",
        "",
        safe,
        flags=re.IGNORECASE,
    )
    lead, sep, tail = prefix_removed.partition("：")
    target_span = lead if sep else prefix_removed
    context = tail if sep else ""

    if not context:
        sentence_parts = re.split(r"[。？！?!.]", prefix_removed, maxsplit=1)
        target_span = sentence_parts[0]
        context = sentence_parts[1] if len(sentence_parts) > 1 else ""
    target_span = _trim_target_span(target_span)
    context = _trim_context_span(context)

    objects = _split_comparison_objects(target_span)
    if len(objects) < 2 and sep:
        objects = _split_comparison_objects(context)
        context = lead
    if len(objects) < 2:
        return []

    context_parts = [_remove_leading_compare_tokens(context)]
    if sep:
        context_parts.append(_remove_object_connectors(target_span, objects))
    shared_context = _clean_variant_text(" ".join(part for part in context_parts if part))
    subqueries: list[str] = []
    for obj in objects[:4]:
        pieces = [obj]
        if shared_context:
            pieces.append(shared_context)
        sq = _clean_variant_text(" ".join(pieces))
        subqueries.append(_unmask_organism_abbrevs(sq))
    return [item for item in subqueries if item]


def _split_comparison_objects(text: str) -> list[str]:
    cleaned = _trim_target_span(_clean_variant_text(_remove_leading_compare_tokens(text)))
    if "：" in cleaned:
        cleaned = cleaned.split("：", 1)[1].strip()
    cleaned = re.sub(r"\b(?:两类|三类|四类)\b", " ", cleaned)
    parts = [item.strip(" ;；。") for item in _COMPARISON_SPLIT_RE.split(cleaned) if item.strip(" ;；。")]
    normalized: list[str] = []
    for part in parts:
        part = re.sub(r"\s*(?:两类|三类|四类)\s+[-A-Za-z0-9\u4e00-\u9fff′'/]+$", "", part).strip()
        if len(part) < 2:
            continue
        if any(token in part for token in ("当前文库", "工程化合成路径", "请分别说明", "请说明", "分别优化")):
            continue
        normalized.append(part)
    return normalized


def _matching_keyword_groups(question: str) -> list[tuple[str, ...]]:
    lowered = question.lower()
    groups: list[tuple[str, ...]] = []
    for keyword in _TITLE_BOOST_KEYWORDS:
        if keyword in lowered:
            groups.append(_KEYWORD_ALIASES.get(keyword, (keyword,)))
    return groups


def _expand_query_aliases(query: str) -> str:
    lowered = query.lower()
    additions: list[str] = []
    seen: set[str] = set()
    for keyword in _TITLE_BOOST_KEYWORDS:
        if keyword not in lowered:
            continue
        for alias in _KEYWORD_ALIASES.get(keyword, (keyword,)):
            if alias in lowered or alias in seen:
                continue
            additions.append(alias)
            seen.add(alias)
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


def _expand_route_pathway_terms(query: str) -> str:
    lowered = query.lower()
    if not any(marker in query for marker in _ROUTE_QUERY_MARKERS):
        return query
    additions: list[str] = []
    seen: set[str] = set()
    for keyword, aliases in _ROUTE_TERM_ALIASES.items():
        if keyword not in lowered:
            continue
        for alias in aliases:
            if alias in lowered or alias in seen:
                continue
            additions.append(alias)
            seen.add(alias)
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


def _mask_organism_abbrevs(text: str) -> str:
    return _ORGANISM_ABBREV_RE.sub(r"\1__\2", text)


def _unmask_organism_abbrevs(text: str) -> str:
    return re.sub(r"([A-Z])__(\w+)", r"\1. \2", text)


def _remove_leading_compare_tokens(text: str) -> str:
    return re.sub(
        r"^(?:文库中|当前文库里|当前文库中|请|比较|对比|相比|compare)\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )


def _remove_object_connectors(text: str, objects: list[str]) -> str:
    cleaned = text
    for obj in objects:
        cleaned = cleaned.replace(obj, " ")
    cleaned = _COMPARISON_SPLIT_RE.sub(" ", cleaned)
    return _clean_variant_text(cleaned)


def _clean_variant_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("：", " ").replace(":", " ")).strip()


def _trim_target_span(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.split(r"[。？！?!.]", cleaned, maxsplit=1)[0]
    cleaned = re.split(r"(?:在当前文库|在文库|在本研究|请分别说明|请说明|分别说明|说明)", cleaned, maxsplit=1)[0]
    return cleaned.strip(" ，,;；")


def _trim_context_span(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^请(?:分别)?说明", "", cleaned)
    cleaned = re.sub(r"^它们", "", cleaned)
    return _clean_variant_text(cleaned)
