from __future__ import annotations

import re

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
}
_GENERIC_ORGANISMS = {"e. coli", "ecoli", "pichia", "yeast", "strain", "host"}
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
)
_EN_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.'+\-]*")
_CJK_PHRASE_RE = re.compile(r"[\u4e00-\u9fff]{2,10}")
_ALIASES = {
    "甲醇诱导": ("methanol induction", "methanol", "induction"),
    "能量利用": ("energy utilization", "energy", "atp"),
    "唾液酸代谢": ("sialic acid metabolism", "sialic acid"),
    "传感器": ("biosensor", "sensor"),
    "启动子": ("promoter",),
    "表达盒拷贝数": ("expression cassette copy number", "copy number", "multi-copy"),
    "表达盒": ("expression cassette",),
    "拷贝数": ("copy number", "multi-copy"),
}


def build_comparison_coverage(
    question: str,
    branches: list[str],
    support_pack: list[SupportItem],
    candidates: list[EvidenceCandidate] | None = None,
) -> ComparisonCoverage:
    del question, candidates
    if not branches:
        return ComparisonCoverage(parse_ok=False, reason="no_branches")

    branch_evidence: list[BranchEvidence] = []
    covered_branches: list[str] = []
    missing_branches: list[str] = []
    allowed_ids: list[str] = []

    for branch in branches:
        assessments = [_score_branch_against_support(branch, item) for item in support_pack]
        best = max(assessments, key=lambda item: (item.confidence, _status_rank(item.status)), default=None)
        if best is None:
            best = BranchEvidence(branch=branch, status="missing", reasons=["no_support_pack"])

        branch_evidence.append(best)
        if best.status in {"direct", "indirect"}:
            covered_branches.append(branch)
            for evidence_id in best.evidence_ids:
                if evidence_id not in allowed_ids:
                    allowed_ids.append(evidence_id)
        else:
            missing_branches.append(branch)

    if not allowed_ids and support_pack:
        for item in support_pack[:3]:
            if item.evidence_id not in allowed_ids:
                allowed_ids.append(item.evidence_id)

    reason = "branch_assignment_complete"
    if all(entry.status == "missing" for entry in branch_evidence):
        reason = "all_branches_missing"
    elif any(entry.status == "indirect" for entry in branch_evidence):
        reason = "indirect_branch_support"
    elif all(entry.status == "direct" for entry in branch_evidence):
        reason = "all_branches_direct"

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
    if "e. coli" in lowered and "E. coli" not in terms:
        terms.append("E. coli")
    elif "ecoli" in lowered and "E. coli" not in terms:
        terms.append("E. coli")

    for raw_token in _EN_TOKEN_RE.findall(normalized):
        token = raw_token.strip(".,;:()[]{}")
        lower_token = token.lower()
        if not token or lower_token in _GENERIC_TERMS:
            continue
        if lower_token == "e.":
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

    filtered = [term for term in terms if not _is_generic_term(term)]
    compacted: list[str] = []
    for term in filtered:
        if (
            any(marker in term for marker in ("和", "及"))
            and any(other != term and other in term for other in filtered)
        ):
            continue
        compacted.append(term)
    return compacted


def score_branch_support(branch: str, item: SupportItem) -> BranchEvidence:
    return _score_branch_against_support(branch, item)


def _score_branch_against_support(branch: str, item: SupportItem) -> BranchEvidence:
    terms = extract_branch_terms(branch)
    haystack = _build_haystack(item)
    matched_terms = [term for term in terms if _term_matches(term, haystack)]
    missing_terms = [term for term in terms if term not in matched_terms]
    entity_matches = [term for term in matched_terms if _is_core_entity(term)]
    specific_matches = [term for term in matched_terms if _is_branch_specific(term)]
    generic_matches = [term for term in matched_terms if term.lower() in _GENERIC_ORGANISMS]

    confidence = min(max(item.support_score, 0.0), 1.5) * 0.2
    confidence += min(len(entity_matches) * 0.32, 0.64)
    confidence += min(len(specific_matches) * 0.22, 0.44)
    confidence += min(len(generic_matches) * 0.08, 0.08)
    if _section_bonus(item.candidate.section) > 0:
        confidence += _section_bonus(item.candidate.section)
    if item.candidate.title and any(_term_matches(term, item.candidate.title.lower()) for term in matched_terms):
        confidence += 0.08
    confidence = min(confidence, 1.0)

    reasons: list[str] = []
    if terms:
        reasons.append(f"branch_terms:{len(terms)}")
    if entity_matches:
        reasons.append("core_entity_overlap")
    if specific_matches:
        reasons.append("branch_specific_overlap")
    if generic_matches:
        reasons.append("generic_organism_overlap")
    if _section_bonus(item.candidate.section) > 0:
        reasons.append("section_bonus")

    status = "missing"
    if terms:
        if entity_matches and specific_matches and len(matched_terms) >= 2:
            status = "direct"
            reasons.append("status:direct")
        elif len(matched_terms) >= 2 and not _only_generic_matches(matched_terms):
            status = "direct"
            reasons.append("status:direct_multi_term")
        elif matched_terms and not _only_generic_matches(matched_terms):
            status = "indirect"
            reasons.append("status:indirect")
        elif generic_matches:
            status = "indirect"
            reasons.append("status:generic_indirect")
    else:
        reasons.append("empty_branch_terms")

    return BranchEvidence(
        branch=branch,
        status=status,
        evidence_ids=[item.evidence_id] if status in {"direct", "indirect"} else [],
        confidence=round(confidence, 4),
        reasons=reasons,
        matched_terms=matched_terms,
        missing_terms=missing_terms,
    )


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


def _only_generic_matches(terms: list[str]) -> bool:
    return bool(terms) and all(term.lower() in _GENERIC_ORGANISMS or _is_generic_term(term) for term in terms)


def _status_rank(status: str) -> int:
    if status == "direct":
        return 2
    if status == "indirect":
        return 1
    return 0


def _section_bonus(section: str) -> float:
    lowered = (section or "").lower()
    if "result" in lowered and "discussion" in lowered:
        return 0.12
    if "result" in lowered:
        return 0.12
    if "discussion" in lowered:
        return 0.08
    return 0.0
