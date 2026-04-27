from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class BranchParseResult:
    branches: list[str] = field(default_factory=list)
    parse_ok: bool = False
    reason: str = ""
    parser_patterns_tried: list[str] = field(default_factory=list)


_GENERIC_BRANCH_TERMS = {
    "策略",
    "机制",
    "作用",
    "用途",
    "角色",
    "方法",
    "路径",
    "差异",
    "区别",
    "方案",
    "一类",
    "另一类",
    "一种",
    "另一种",
    "两类",
    "两种",
    "strategy",
    "mechanism",
    "role",
    "usage",
    "function",
}

_INLINE_PATTERNS = [
    (
        "zh_compare_with_two_item_suffix",
        re.compile(
            r"^(?:请)?(?:比较|对比)\s*(?:文库中|当前文库中)?\s*(.+?)\s*[和与]\s*(.+?)\s*两(?:种|类|项)(?:[^？?。.]*)[？?。.]?$"
        ),
    ),
    (
        "zh_compare_with_difference_suffix",
        re.compile(
            r"^(?:请)?(?:比较|对比)\s*(?:文库中|当前文库中)?\s*(.+?)\s*[和与]\s*(.+?)\s*(?:之间的)?(?:异同|差异|区别|比较)(?:$|[？?。.,，])"
        ),
    ),
    (
        "zh_compare_basic",
        re.compile(r"比较\s+(.+?)\s+[和与]\s+(.+?)(?:的|在|有何|有什么|差异|区别|作用|机制|用途|角色|$|[？?。.,，])"),
    ),
    ("vs_form", re.compile(r"(.+?)\s+vs\.?\s+(.+?)(?:$|[？?。.,，])", re.IGNORECASE)),
    ("compare_and", re.compile(r"compare\s+(.+?)\s+and\s+(.+?)(?:$|[?.。,，])", re.IGNORECASE)),
    (
        "comparison_between",
        re.compile(r"comparison between\s+(.+?)\s+and\s+(.+?)(?:$|[?.。,，])", re.IGNORECASE),
    ),
    ("respectively", re.compile(r"分别是\s+(.+?)\s+[和与]\s+(.+?)(?:$|[？?。.,，])")),
]

_TWO_CLASS_PATTERNS = [
    ("one_class_another_class", re.compile(r"一类[是为]\s*(.+?)，\s*另一类[是为]\s*(.+?)(?:$|[？?。])")),
    ("one_kind_another_kind", re.compile(r"一种[是为]\s*(.+?)，\s*另一种[是为]\s*(.+?)(?:$|[？?。])")),
    (
        "two_classes_with_label",
        re.compile(r"两类[^：:]*[:：]\s*一类[是为]\s*(.+?)，\s*另一类[是为]\s*(.+?)(?:$|[？?。])"),
    ),
    (
        "two_kinds_with_label",
        re.compile(r"两种[^：:]*[:：]\s*一种[是为]\s*(.+?)，\s*另一种[是为]\s*(.+?)(?:$|[？?。])"),
    ),
]


def parse_comparison_branches(question: str) -> BranchParseResult:
    cleaned = re.sub(r"\s+", " ", question or "").strip().strip("？?。.")
    if not cleaned:
        return BranchParseResult(reason="empty_question")
    if not _looks_like_comparison_question(cleaned):
        return BranchParseResult(reason="not_comparison_intent")

    patterns_tried: list[str] = []
    first_invalid_result: BranchParseResult | None = None
    for name, pattern in _TWO_CLASS_PATTERNS:
        patterns_tried.append(name)
        match = pattern.search(cleaned)
        if match:
            result = _build_result(match.group(1), match.group(2), patterns_tried)
            if result.parse_ok:
                return result
            if first_invalid_result is None:
                first_invalid_result = result

    for name, pattern in _INLINE_PATTERNS:
        patterns_tried.append(name)
        match = pattern.search(cleaned)
        if match:
            result = _build_result(match.group(1), match.group(2), patterns_tried)
            if result.parse_ok:
                return result
            if first_invalid_result is None:
                first_invalid_result = result

    if first_invalid_result is not None:
        return BranchParseResult(
            reason=first_invalid_result.reason,
            parser_patterns_tried=patterns_tried,
        )
    return BranchParseResult(reason="no_supported_pattern", parser_patterns_tried=patterns_tried)


def clean_branch_text(value: str) -> str:
    branch = (value or "").strip(" ，,：:；;。.")
    branch = re.sub(
        r"^(比较|对比|请比较|请对比|文库中|当前文库中|当前知识库中|文库里)\s*",
        "",
        branch,
        flags=re.IGNORECASE,
    )
    branch = re.sub(r"^(其中|分别|一类|另一类|一种|另一种)\s*", "", branch).strip(" ，,：:；;。.")
    branch = re.sub(
        r"(?:两(?:种|类|项).{0,20}?(?:策略|研究|效应|方法|方案)|的异同|的差异|的区别|之间的差异|之间的异同|的比较)$",
        "",
        branch,
    ).strip(" ，,：:；;。.")
    branch = re.sub(r"的$", "", branch).strip(" ，,：:；;。.")
    return branch


def _build_result(left: str, right: str, patterns_tried: list[str]) -> BranchParseResult:
    branches = [_normalize_branch(left), _normalize_branch(right)]
    if not branches[0] or not branches[1]:
        return BranchParseResult(reason="empty_branch", parser_patterns_tried=list(patterns_tried))
    if branches[0] == branches[1]:
        return BranchParseResult(reason="duplicate_branch", parser_patterns_tried=list(patterns_tried))
    for branch in branches:
        invalid_reason = _validate_branch(branch)
        if invalid_reason:
            return BranchParseResult(reason=invalid_reason, parser_patterns_tried=list(patterns_tried))
    return BranchParseResult(branches=branches, parse_ok=True, reason="ok", parser_patterns_tried=list(patterns_tried))


def _normalize_branch(value: str) -> str:
    branch = clean_branch_text(value)
    branch = re.sub(r"^(compare|comparison between)\s+", "", branch, flags=re.IGNORECASE)
    branch = re.sub(r"(的两种角色|的两类角色|的角色|的策略|的机制|的作用|的用途)$", "", branch).strip(" ，,：:；;。.")
    return branch


def _validate_branch(branch: str) -> str:
    normalized = branch.lower().strip()
    if not normalized:
        return "empty_branch"
    if normalized in _GENERIC_BRANCH_TERMS:
        return "generic_branch"
    if len(branch) > 80 or len(normalized.encode("utf-8")) > 120:
        return "branch_too_long"
    if re.fullmatch(r"[a-z0-9_.-]{1,4}", normalized):
        return ""
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalized)
    if len("".join(tokens)) < 2:
        return "branch_too_short"
    if len(tokens) == 1 and tokens[0] in _GENERIC_BRANCH_TERMS:
        return "generic_branch"
    if all(token in _GENERIC_BRANCH_TERMS for token in tokens):
        return "generic_branch"
    return ""


def _looks_like_comparison_question(question: str) -> bool:
    return bool(
        re.search(
            r"(?:\bvs\.?\b|\bcompare\b|\bcomparison\b|比较|对比|异同|差异|区别|两种|两类|两项|一类|另一类|一种|另一种)",
            question,
            re.IGNORECASE,
        )
    )
