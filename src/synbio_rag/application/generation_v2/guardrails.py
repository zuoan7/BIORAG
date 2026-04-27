from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .models import SupportItem


@dataclass
class ExistenceQuestionSignal:
    is_existence_question: bool
    reason: str
    matched_patterns: list[str] = field(default_factory=list)
    target_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ExistenceSupportAssessment:
    support_status: str
    reason: str
    matched_core_terms: list[str] = field(default_factory=list)
    missing_core_terms: list[str] = field(default_factory=list)
    weak_signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_LIBRARY_PATTERNS = [
    (r"文库中", "cn:library_in"),
    (r"当前文库[中里]", "cn:current_library"),
    (r"知识库中", "cn:knowledge_base"),
    (r"\blibrary\b", "en:library"),
    (r"\bknowledge base\b", "en:knowledge_base"),
]

_EXISTENCE_PATTERNS = [
    (r"是否有关于", "cn:has_about"),
    (r"有没有关于", "cn:any_about"),
    (r"是否有", "cn:has"),
    (r"有没有", "cn:any"),
    (r"是否存在", "cn:exists"),
    (r"是否包含", "cn:contains"),
    (r"是否提供", "cn:provides"),
    (r"\bis there any\b", "en:is_there_any"),
    (r"\bdoes the library contain\b", "en:library_contains"),
    (r"\bis there evidence for\b", "en:evidence_for"),
]

_INSTRUCTION_PATTERNS = [
    (r"如果没有.*证据不足", "cn:if_not_insufficient"),
    (r"如果没有，请明确说明证据不足", "cn:explicit_insufficient"),
    (r"不要扩展到文库外", "cn:no_external_expansion"),
    (r"请基于文库内容回答", "cn:library_only_answer"),
    (r"\bif not, say insufficient evidence\b", "en:if_not_insufficient"),
    (r"\bdo not use external knowledge\b", "en:no_external_knowledge"),
    (r"\bonly based on (?:the )?(?:library|knowledge base)\b", "en:library_only"),
]

_MECHANISM_PATTERNS = [
    r"是否调控",
    r"是否促进",
    r"是否抑制",
    r"是否提高",
    r"是否降低",
    r"是否通过",
    r"是否影响",
    r"是否激活",
    r"是否改善",
    r"是否介导",
]

_COMPARISON_HINTS = [r"比较", r"\bcompare\b", r"\bcomparison\b", r"\bvs\.?\b", r"对比"]

_RESOURCE_TERMS = [
    "数据",
    "详细方案",
    "系统综述",
    "临床试验",
    "发酵工艺",
    "工艺",
    "策略",
    "报告",
    "protocol",
    "review",
    "trial",
    "data",
    "scheme",
    "strategy",
]

_CORE_PHRASES = [
    "工业规模",
    "industrial scale",
    "industrial-scale",
    "溶解氧",
    "dissolved oxygen",
    "control strategy",
    "控制策略",
    "详细方案",
    "systematic review",
    "系统综述",
    "临床试验",
    "clinical trial",
    "phase iii",
    "iii 期",
    "发酵工艺",
    "fermentation process",
    "protocol",
    "review",
    "trial",
    "data",
]

_GENERIC_TERMS = {
    "文库",
    "当前文库",
    "知识库",
    "资料",
    "数据",
    "方案",
    "详细方案",
    "系统综述",
    "临床试验",
    "工艺",
    "策略",
    "报告",
    "protocol",
    "review",
    "trial",
    "data",
    "scheme",
    "strategy",
}

_UPPER_TOKEN_PATTERN = re.compile(r"\b[A-Z][A-Z0-9-]{1,}\b")
_MIXED_TOKEN_PATTERN = re.compile(r"\b(?:mRNA|COVID|CAR-T|DLBCL|DO|PHA|III)\b", re.IGNORECASE)
_TERM_ALIASES = {
    "工业规模": ["industrial scale", "industrial-scale", "large-scale"],
    "溶解氧": ["dissolved oxygen", " do ", "(do)", "do setpoint", "do control"],
    "控制策略": ["control strategy", "control strategies", "control scheme"],
    "详细方案": ["detailed scheme", "detailed protocol", "detailed strategy", "detailed plan"],
    "发酵工艺": ["fermentation process", "fermentation processes", "fermentation"],
    "系统综述": ["systematic review", "review"],
    "临床试验": ["clinical trial", "clinical trials", "trial"],
    "iii 期": ["phase iii", "phase 3", "iii"],
}


def detect_existence_question(question: str) -> ExistenceQuestionSignal:
    cleaned = _normalize_text(question)
    lowered = cleaned.lower()
    matched_patterns: list[str] = []

    if any(re.search(pattern, cleaned, re.IGNORECASE) for pattern in _COMPARISON_HINTS):
        return ExistenceQuestionSignal(
            is_existence_question=False,
            reason="comparison_question",
            target_terms=_extract_target_terms(question),
        )

    library_hit = _collect_pattern_hits(cleaned, _LIBRARY_PATTERNS, matched_patterns)
    existence_hit = _collect_pattern_hits(cleaned, _EXISTENCE_PATTERNS, matched_patterns)
    instruction_hit = _collect_pattern_hits(cleaned, _INSTRUCTION_PATTERNS, matched_patterns)

    if not (library_hit or existence_hit or instruction_hit):
        return ExistenceQuestionSignal(
            is_existence_question=False,
            reason="no_existence_signal",
            target_terms=_extract_target_terms(question),
        )

    if (
        not library_hit
        and not instruction_hit
        and any(re.search(pattern, cleaned) for pattern in _MECHANISM_PATTERNS)
    ):
        return ExistenceQuestionSignal(
            is_existence_question=False,
            reason="mechanism_predicate_only",
            matched_patterns=matched_patterns,
            target_terms=_extract_target_terms(question),
        )

    resource_hit = [term for term in _RESOURCE_TERMS if term.lower() in lowered]
    if resource_hit:
        matched_patterns.extend(f"resource:{term}" for term in resource_hit)

    is_existence = bool((library_hit and existence_hit) or instruction_hit or ("关于" in cleaned and existence_hit))
    reason = "library_existence_signal" if is_existence else "insufficient_existence_signal"
    return ExistenceQuestionSignal(
        is_existence_question=is_existence,
        reason=reason,
        matched_patterns=list(dict.fromkeys(matched_patterns)),
        target_terms=_extract_target_terms(question),
    )


def evaluate_existence_support(
    question: str,
    support_pack: list[SupportItem],
    candidates: list[object] | None = None,
) -> ExistenceSupportAssessment:
    del candidates
    if not support_pack:
        return ExistenceSupportAssessment(
            support_status="none",
            reason="no_support_pack",
            weak_signals=["empty_support_pack"],
        )

    core_terms = _extract_target_terms(question)
    support_text = _normalize_text(
        " ".join(
            " ".join(
                [
                    item.candidate.title or "",
                    item.candidate.section or "",
                    item.candidate.text or "",
                    " ".join(str(value) for value in item.candidate.metadata.values()),
                ]
            )
            for item in support_pack
        )
    ).lower()

    matched_core_terms = [term for term in core_terms if term and _term_in_text(term, support_text)]
    missing_core_terms = [term for term in core_terms if term not in matched_core_terms]
    weak_signals: list[str] = []

    if not matched_core_terms:
        weak_signals.append("no_core_term_match")
    if missing_core_terms:
        weak_signals.append("missing_core_terms")

    entity_terms = [term for term in core_terms if _is_entity_term(term)]
    qualifier_terms = [term for term in core_terms if term not in entity_terms]
    matched_entities = [term for term in entity_terms if term in matched_core_terms]
    matched_qualifiers = [term for term in qualifier_terms if term in matched_core_terms]

    if entity_terms and not matched_entities:
        weak_signals.append("missing_primary_entity")
    if qualifier_terms and not matched_qualifiers:
        weak_signals.append("missing_key_qualifier")
    if any(_cross_topic_signal(item, entity_terms) for item in support_pack):
        weak_signals.append("cross_topic_support")

    if not core_terms:
        return ExistenceSupportAssessment(
            support_status="weak",
            reason="no_extractable_core_terms",
            weak_signals=["no_extractable_core_terms"],
        )

    matched_ratio = len(matched_core_terms) / len(core_terms)
    if matched_ratio >= 0.6 and (not entity_terms or matched_entities) and matched_qualifiers:
        return ExistenceSupportAssessment(
            support_status="strong",
            reason="core_terms_covered",
            matched_core_terms=matched_core_terms,
            missing_core_terms=missing_core_terms,
        )

    return ExistenceSupportAssessment(
        support_status="weak",
        reason="core_terms_not_sufficiently_covered",
        matched_core_terms=matched_core_terms,
        missing_core_terms=missing_core_terms,
        weak_signals=list(dict.fromkeys(weak_signals)),
    )


def _collect_pattern_hits(text: str, patterns: list[tuple[str, str]], matched_patterns: list[str]) -> bool:
    matched = False
    for pattern, label in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            matched_patterns.append(label)
            matched = True
    return matched


def _extract_target_terms(question: str) -> list[str]:
    lowered = _normalize_text(question).lower()
    terms: list[str] = []
    for phrase in _CORE_PHRASES:
        if phrase.lower() in lowered:
            terms.append(phrase)

    for token in _UPPER_TOKEN_PATTERN.findall(question):
        terms.append(token)
    for token in _MIXED_TOKEN_PATTERN.findall(question):
        terms.append(token)

    return list(dict.fromkeys(term.strip() for term in terms if term.strip()))


def _is_entity_term(term: str) -> bool:
    lowered = term.lower()
    if term in _GENERIC_TERMS:
        return False
    if re.fullmatch(r"[A-Z0-9-]{2,}", term):
        return True
    if re.fullmatch(r"(?:mRNA|COVID|CAR-T|DLBCL|PHA|DO|III)", term, re.IGNORECASE):
        return True
    return lowered not in {phrase.lower() for phrase in _CORE_PHRASES if " " in phrase or any("\u4e00" <= ch <= "\u9fff" for ch in phrase)}


def _cross_topic_signal(item: SupportItem, entity_terms: list[str]) -> bool:
    if not entity_terms:
        return False
    text = _normalize_text(item.candidate.text).lower()
    lower_entities = {term.lower() for term in entity_terms}
    unrelated_hits = [term for term in ("l-精氨酸", "arginine", "lysine", "dextranase") if term.lower() in text]
    return bool(unrelated_hits) and not any(term in text for term in lower_entities)


def _term_in_text(term: str, text: str) -> bool:
    normalized = term.lower().strip()
    if not normalized:
        return False
    aliases = _TERM_ALIASES.get(term, []) + _TERM_ALIASES.get(normalized, [])
    for alias in aliases:
        alias_lower = alias.lower()
        if alias_lower.strip() and alias_lower in text:
            return True
    if re.fullmatch(r"[a-z0-9-]{2,}", normalized):
        return bool(re.search(rf"\b{re.escape(normalized)}\b", text))
    return normalized in text


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()
