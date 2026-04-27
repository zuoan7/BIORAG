from __future__ import annotations

import re
from dataclasses import dataclass

from .models import BranchEvidence, ComparisonCoverage, EvidenceCandidate, SupportItem

_GENERIC_TERMS = {
    "调控",
    "作用",
    "机制",
    "用途",
    "策略",
    "角色",
    "差异",
    "比较",
    "对比",
    "一类",
    "另一类",
    "一种",
    "另一种",
    "两类",
    "两种",
    "strategy",
    "mechanism",
    "role",
    "difference",
    "comparison",
    "expression",
    "promoter",
}
_GENERIC_DIRECT_BLOCKERS = {
    "甲醇",
    "pichia",
    "e. coli",
    "promoter",
    "expression",
    "提升表达",
}
_GENERIC_ORGANISMS = {"e. coli", "ecoli", "pichia", "yeast", "strain", "host"}
_LOW_QUALITY_SECTION_TOKENS = ("reference", "bibliograph", "acknowledg", "author", "title", "content")
_CORE_PHRASES = (
    "唾液酸代谢",
    "天然调控",
    "工程应用",
    "甲醇诱导",
    "能量利用",
    "表达盒拷贝数",
    "表达盒",
    "拷贝数",
    "启动子",
    "调控表达盒",
    "传感器",
    "biosensor",
    "promoter",
    "copy number",
    "methanol induction",
    "energy utilization",
    "dissolved oxygen",
    "industrial scale",
    "sialoregulon",
    "multi-copy",
    "expression cassette",
)
_DROP_PREFIXES = ("被改造成", "直接增加", "优化", "另一类是", "一类是", "另一种是", "一种是", "分别是")
_EN_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.'+\-]*")
_CJK_PHRASE_RE = re.compile(r"[\u4e00-\u9fff]{2,12}")
_ALIASES = {
    "甲醇诱导": ("methanol induction", "methanol", "induction"),
    "能量利用": ("energy utilization", "energy", "atp", "nadh", "our", "rmeoh", "methanol consumption"),
    "唾液酸代谢": ("sialic acid metabolism", "sialic acid", "catabolism", "sialoregulon"),
    "传感器": ("biosensor", "sensor"),
    "启动子": ("promoter",),
    "表达盒拷贝数": ("expression cassette copy number", "copy number", "multi-copy"),
    "表达盒": ("expression cassette",),
    "拷贝数": ("copy number", "multi-copy"),
    "天然调控": ("regulation", "repressor", "derepresses"),
    "工程应用": ("engineering", "engineered", "application"),
}


@dataclass
class _BranchAssessment:
    evidence_id: str
    status: str
    confidence: float
    matched_terms: list[str]
    missing_terms: list[str]
    reasons: list[str]
    support_score: float
    section: str


def build_comparison_coverage(
    question: str,
    branches: list[str],
    support_pack: list[SupportItem],
    candidates: list[EvidenceCandidate] | None = None,
) -> ComparisonCoverage:
    del question, candidates
    if not branches:
        return ComparisonCoverage(parse_ok=False, reason="no_valid_branches")
    if not support_pack:
        return ComparisonCoverage(parse_ok=True, branches=list(branches), reason="empty_support_pack")

    branch_evidence: list[BranchEvidence] = []
    allowed_ids: list[str] = []
    support_by_id = {item.evidence_id: item for item in support_pack}

    for branch in branches:
        assessments = [_score_branch_against_support(branch, item) for item in support_pack]
        primary = _pick_primary_assessment(branch, assessments)
        secondary = _pick_secondary_assessments(branch, primary, assessments)
        branch_entry = _make_branch_evidence(branch, primary, secondary)
        branch_evidence.append(branch_entry)
        for evidence_id in branch_entry.evidence_ids:
            if evidence_id in support_by_id and evidence_id not in allowed_ids:
                allowed_ids.append(evidence_id)

    _apply_shared_evidence_penalty(branch_evidence)

    covered_branches = [entry.branch for entry in branch_evidence if entry.status in {"direct", "indirect"}]
    missing_branches = [entry.branch for entry in branch_evidence if entry.status == "missing"]

    if not allowed_ids and support_pack:
        for item in support_pack[:3]:
            if item.evidence_id not in allowed_ids:
                allowed_ids.append(item.evidence_id)

    reason = _coverage_reason(branch_evidence, allowed_ids)
    return ComparisonCoverage(
        parse_ok=True,
        branches=list(branches),
        branch_evidence=branch_evidence,
        covered_branches=covered_branches,
        missing_branches=missing_branches,
        allowed_citation_evidence_ids=allowed_ids,
        reason=reason,
    )


def extract_branch_terms(branch: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", branch or "").strip()
    if not normalized:
        return []

    terms: list[str] = []
    lowered = normalized.lower()
    for phrase in _CORE_PHRASES:
        if phrase.lower() in lowered:
            terms.append(phrase)
    if ("e. coli" in lowered or "ecoli" in lowered) and "E. coli" not in terms:
        terms.append("E. coli")

    for raw_token in _EN_TOKEN_RE.findall(normalized):
        token = raw_token.strip(".,;:()[]{}")
        lower_token = token.lower()
        if not token or lower_token in _GENERIC_TERMS or lower_token == "e.":
            continue
        if lower_token == "coli" and "E. coli" in terms:
            continue
        if len(token) == 1 and token.isalpha():
            continue
        if token not in terms:
            terms.append(token)

    for phrase in _CJK_PHRASE_RE.findall(normalized):
        if phrase in _GENERIC_TERMS or len(phrase) < 2:
            continue
        if phrase not in terms:
            terms.append(phrase)

    filtered = [term for term in terms if not _is_generic_term(term) and not _drop_term(term)]
    compacted: list[str] = []
    for term in filtered:
        if (
            term not in _CORE_PHRASES
            and not _is_core_entity(term)
            and any(other != term and term in other and len(other) > len(term) for other in filtered)
        ):
            continue
        compacted.append(term)
    return compacted


def score_branch_support(branch: str, item: SupportItem) -> BranchEvidence:
    assessment = _score_branch_against_support(branch, item)
    evidence_ids = [assessment.evidence_id] if assessment.status in {"direct", "indirect"} else []
    return BranchEvidence(
        branch=branch,
        status=assessment.status,
        evidence_ids=evidence_ids,
        primary_evidence_ids=evidence_ids,
        secondary_evidence_ids=[],
        confidence=assessment.confidence,
        reasons=list(assessment.reasons),
        matched_terms=list(assessment.matched_terms),
        missing_terms=list(assessment.missing_terms),
    )


def _score_branch_against_support(branch: str, item: SupportItem) -> _BranchAssessment:
    terms = extract_branch_terms(branch)
    haystack = _build_haystack(item)
    matched_terms = [term for term in terms if _term_matches(term, haystack)]
    missing_terms = [term for term in terms if term not in matched_terms]
    entity_matches = [term for term in matched_terms if _is_core_entity(term)]
    specific_matches = [term for term in matched_terms if _is_branch_specific(term)]
    strong_specific_matches = [term for term in specific_matches if term.lower() not in _GENERIC_DIRECT_BLOCKERS]
    generic_matches = [term for term in matched_terms if term.lower() in _GENERIC_ORGANISMS]

    confidence = min(max(item.support_score, 0.0), 1.5) * 0.18
    confidence += min(len(entity_matches) * 0.18, 0.36)
    confidence += min(len(strong_specific_matches) * 0.16, 0.48)
    confidence += _section_bonus(item.candidate.section)
    if item.candidate.title and any(_term_matches(term, item.candidate.title.lower()) for term in matched_terms):
        confidence += 0.08
    confidence = min(confidence, 1.0)

    reasons: list[str] = []
    if terms:
        reasons.append(f"branch_terms:{len(terms)}")
    if entity_matches:
        reasons.append("core_entity_overlap")
    if strong_specific_matches:
        reasons.append("branch_specific_overlap")
    if generic_matches:
        reasons.append("generic_organism_overlap")
    if _section_bonus(item.candidate.section) > 0:
        reasons.append("section_bonus")

    status = "missing"
    if terms:
        if _has_strong_direct_signal(branch, matched_terms, entity_matches, strong_specific_matches, haystack):
            status = "direct"
            reasons.append("status:direct")
        elif _has_indirect_signal(matched_terms, entity_matches, strong_specific_matches):
            status = "indirect"
            reasons.append("status:indirect")
        elif generic_matches:
            status = "indirect"
            reasons.append("status:generic_indirect")
    else:
        reasons.append("empty_branch_terms")

    return _BranchAssessment(
        evidence_id=item.evidence_id,
        status=status,
        confidence=round(confidence, 4),
        matched_terms=matched_terms,
        missing_terms=missing_terms,
        reasons=reasons,
        support_score=item.support_score,
        section=item.candidate.section,
    )


def _pick_primary_assessment(branch: str, assessments: list[_BranchAssessment]) -> _BranchAssessment:
    if not assessments:
        return _BranchAssessment(
            evidence_id="",
            status="missing",
            confidence=0.0,
            matched_terms=[],
            missing_terms=extract_branch_terms(branch),
            reasons=["no_support_pack"],
            support_score=0.0,
            section="",
        )
    return max(assessments, key=lambda item: (_status_rank(item.status), item.confidence, item.support_score))


def _pick_secondary_assessments(
    branch: str,
    primary: _BranchAssessment,
    assessments: list[_BranchAssessment],
) -> list[_BranchAssessment]:
    del branch
    secondaries: list[_BranchAssessment] = []
    for candidate in sorted(
        assessments,
        key=lambda item: (_status_rank(item.status), item.confidence, item.support_score),
        reverse=True,
    ):
        if candidate.evidence_id == primary.evidence_id:
            continue
        if not _qualifies_secondary(candidate):
            continue
        secondaries.append(candidate)
        if len(secondaries) >= 2:
            break
    return secondaries


def _make_branch_evidence(
    branch: str,
    primary: _BranchAssessment,
    secondary: list[_BranchAssessment],
) -> BranchEvidence:
    primary_ids = [primary.evidence_id] if primary.status in {"direct", "indirect"} and primary.evidence_id else []
    secondary_ids = [entry.evidence_id for entry in secondary]
    evidence_ids = primary_ids + [entry_id for entry_id in secondary_ids if entry_id not in primary_ids]
    reasons = list(primary.reasons)
    if secondary_ids:
        reasons.append(f"secondary_evidence:{len(secondary_ids)}")
    return BranchEvidence(
        branch=branch,
        status=primary.status,
        evidence_ids=evidence_ids,
        primary_evidence_ids=primary_ids,
        secondary_evidence_ids=secondary_ids,
        confidence=primary.confidence,
        reasons=reasons,
        matched_terms=list(primary.matched_terms),
        missing_terms=list(primary.missing_terms),
    )


def _apply_shared_evidence_penalty(branch_evidence: list[BranchEvidence]) -> None:
    primary_to_indexes: dict[str, list[int]] = {}
    for index, entry in enumerate(branch_evidence):
        if entry.status != "direct" or len(entry.primary_evidence_ids) != 1:
            continue
        primary_to_indexes.setdefault(entry.primary_evidence_ids[0], []).append(index)

    for indexes in primary_to_indexes.values():
        if len(indexes) < 2:
            continue
        strongest_index = max(indexes, key=lambda idx: len(branch_evidence[idx].matched_terms))
        for index in indexes:
            entry = branch_evidence[index]
            entry.reasons.extend(["shared_evidence_across_branches", "limited_independent_branch_support"])
            if index == strongest_index and _has_strong_branch_term_combo(entry.matched_terms):
                continue
            entry.status = "indirect"
            if "status:downgraded_shared_primary" not in entry.reasons:
                entry.reasons.append("status:downgraded_shared_primary")


def _coverage_reason(branch_evidence: list[BranchEvidence], allowed_ids: list[str]) -> str:
    if not branch_evidence:
        return "no_valid_branches"
    statuses = {entry.status for entry in branch_evidence}
    if all(entry.status == "missing" for entry in branch_evidence):
        return "all_branches_missing"
    if any("shared_evidence_across_branches" in entry.reasons for entry in branch_evidence):
        return "shared_evidence_limited_comparison"
    if statuses == {"direct"}:
        return "all_branches_direct"
    if any(entry.status == "indirect" for entry in branch_evidence):
        return "indirect_branch_support"
    if allowed_ids:
        return "branch_assignment_complete"
    return "limited_context_only"


def _build_haystack(item: SupportItem) -> str:
    return " ".join(
        part
        for part in [
            (item.candidate.title or "").lower(),
            (item.candidate.section or "").lower(),
            (item.candidate.text or "").lower(),
            " ".join(str(value).lower() for value in item.candidate.metadata.values()),
        ]
        if part
    )


def _term_matches(term: str, haystack: str) -> bool:
    lowered = term.lower()
    if lowered in haystack:
        return True
    for alias in _ALIASES.get(term, ()):
        if alias.lower() in haystack:
            return True
    if lowered == "e. coli":
        return "e. coli" in haystack or "ecoli" in haystack
    if lowered == "neu5ac":
        return "neu5ac" in haystack or "n-acetylneuraminic acid" in haystack
    if lowered == "dissolved oxygen":
        return "dissolved oxygen" in haystack or re.search(r"\bdo\b", haystack) is not None
    if lowered == "methanol induction":
        return "methanol induction" in haystack or ("methanol" in haystack and "induction" in haystack)
    if lowered == "industrial scale":
        return "industrial scale" in haystack or "industrial-scale" in haystack or "large-scale" in haystack
    if lowered == "copy number":
        return "copy number" in haystack or "multi-copy" in haystack
    return False


def _has_strong_direct_signal(
    branch: str,
    matched_terms: list[str],
    entity_matches: list[str],
    strong_specific_matches: list[str],
    haystack: str,
) -> bool:
    matched = set(matched_terms)
    branch_terms = set(extract_branch_terms(branch))
    if {"甲醇诱导", "能量利用"} & branch_terms:
        if "甲醇诱导" in matched and "能量利用" in matched:
            return True
    if "AOX1" in branch_terms:
        has_promoter = "启动子" in matched or "promoter" in matched
        has_copy = bool({"表达盒拷贝数", "拷贝数", "expression cassette", "copy number", "multi-copy"} & matched)
        if "AOX1" in matched and has_promoter and has_copy:
            return True
    if "Neu5Ac" in branch_terms and "传感器" in matched:
        return True
    if "E. coli" in branch_terms and "唾液酸代谢" in matched and "E. coli" in matched:
        return True
    if entity_matches and strong_specific_matches and len(set(entity_matches + strong_specific_matches)) >= 2:
        return True
    if len(strong_specific_matches) >= 3 and _branch_focused_haystack(strong_specific_matches, haystack):
        return True
    return False


def _has_indirect_signal(
    matched_terms: list[str],
    entity_matches: list[str],
    strong_specific_matches: list[str],
) -> bool:
    if entity_matches and strong_specific_matches:
        return True
    if len(strong_specific_matches) >= 1:
        return True
    informative = [term for term in matched_terms if term.lower() not in _GENERIC_DIRECT_BLOCKERS]
    return bool(informative)


def _qualifies_secondary(candidate: _BranchAssessment) -> bool:
    if candidate.status == "missing":
        return False
    if any(token in (candidate.section or "").lower() for token in _LOW_QUALITY_SECTION_TOKENS):
        return False
    informative_matches = [term for term in candidate.matched_terms if term.lower() not in _GENERIC_DIRECT_BLOCKERS]
    if not informative_matches:
        return False
    if candidate.support_score <= 0.0 and candidate.confidence < 0.45:
        return False
    return candidate.confidence >= 0.35 or candidate.support_score >= 1.0


def _branch_focused_haystack(terms: list[str], haystack: str) -> bool:
    return sum(1 for term in terms if _term_matches(term, haystack)) >= 2


def _has_strong_branch_term_combo(matched_terms: list[str]) -> bool:
    matched = set(matched_terms)
    if "AOX1" in matched and (
        "启动子" in matched or "promoter" in matched
    ) and bool({"表达盒拷贝数", "拷贝数", "expression cassette", "copy number", "multi-copy"} & matched):
        return True
    if "甲醇诱导" in matched and "能量利用" in matched:
        return True
    if "Neu5Ac" in matched and "传感器" in matched:
        return True
    return len([term for term in matched_terms if term.lower() not in _GENERIC_DIRECT_BLOCKERS]) >= 3


def _is_core_entity(term: str) -> bool:
    lowered = term.lower()
    if lowered in _GENERIC_ORGANISMS:
        return False
    if any(ch.isdigit() for ch in term):
        return True
    if any(ch.isupper() for ch in term):
        return True
    if "." in term or "-" in term or "′" in term or "'" in term:
        return True
    return False


def _is_branch_specific(term: str) -> bool:
    lowered = term.lower()
    return not _is_generic_term(term) and lowered not in _GENERIC_ORGANISMS


def _is_generic_term(term: str) -> bool:
    return term.lower() in _GENERIC_TERMS or term in _GENERIC_TERMS


def _drop_term(term: str) -> bool:
    return term.startswith(_DROP_PREFIXES)


def _status_rank(status: str) -> int:
    if status == "direct":
        return 2
    if status == "indirect":
        return 1
    return 0


def _section_bonus(section: str) -> float:
    lowered = (section or "").lower()
    if any(token in lowered for token in _LOW_QUALITY_SECTION_TOKENS):
        return -0.1
    if "result" in lowered and "discussion" in lowered:
        return 0.12
    if "result" in lowered:
        return 0.12
    if "discussion" in lowered:
        return 0.08
    return 0.0
