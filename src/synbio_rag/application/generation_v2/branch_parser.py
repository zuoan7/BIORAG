from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class BranchParseResult:
    branches: list[str] = field(default_factory=list)
    parse_ok: bool = False
    reason: str = ""


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
    re.compile(r"比较\s+(.+?)\s+[和与]\s+(.+?)(?:的|在|有何|有什么|差异|区别|作用|机制|用途|角色|$|[？?。.,，])"),
    re.compile(r"(.+?)\s+vs\.?\s+(.+?)(?:$|[？?。.,，])", re.IGNORECASE),
    re.compile(r"compare\s+(.+?)\s+and\s+(.+?)(?:$|[?.。,，])", re.IGNORECASE),
    re.compile(r"comparison between\s+(.+?)\s+and\s+(.+?)(?:$|[?.。,，])", re.IGNORECASE),
    re.compile(r"分别是\s+(.+?)\s+[和与]\s+(.+?)(?:$|[？?。.,，])"),
]

_TWO_CLASS_PATTERNS = [
    re.compile(r"一类[是为]\s*(.+?)，\s*另一类[是为]\s*(.+?)(?:$|[？?。])"),
    re.compile(r"一种[是为]\s*(.+?)，\s*另一种[是为]\s*(.+?)(?:$|[？?。])"),
    re.compile(r"两类[^：:]*[:：]\s*一类[是为]\s*(.+?)，\s*另一类[是为]\s*(.+?)(?:$|[？?。])"),
    re.compile(r"两种[^：:]*[:：]\s*一种[是为]\s*(.+?)，\s*另一种[是为]\s*(.+?)(?:$|[？?。])"),
]


def parse_comparison_branches(question: str) -> BranchParseResult:
    cleaned = re.sub(r"\s+", " ", question or "").strip().strip("？?。.")
    if not cleaned:
        return BranchParseResult(reason="empty_question")

    for pattern in _TWO_CLASS_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            result = _build_result(match.group(1), match.group(2))
            if result.parse_ok:
                return result

    for pattern in _INLINE_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            result = _build_result(match.group(1), match.group(2))
            if result.parse_ok:
                return result

    return BranchParseResult(reason="no_supported_pattern")


def _build_result(left: str, right: str) -> BranchParseResult:
    branches = [_normalize_branch(left), _normalize_branch(right)]
    if not branches[0] or not branches[1]:
        return BranchParseResult(reason="empty_branch")
    if branches[0] == branches[1]:
        return BranchParseResult(reason="duplicate_branch")
    for branch in branches:
        invalid_reason = _validate_branch(branch)
        if invalid_reason:
            return BranchParseResult(reason=invalid_reason)
    return BranchParseResult(branches=branches, parse_ok=True, reason="ok")


def _normalize_branch(value: str) -> str:
    branch = (value or "").strip(" ，,：:；;。.")
    branch = re.sub(r"^(比较|对比|compare|comparison between)\s+", "", branch, flags=re.IGNORECASE)
    branch = re.sub(r"^(其中|分别|一类|另一类|一种|另一种)\s*", "", branch).strip(" ，,：:；;。.")
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
    return ""
