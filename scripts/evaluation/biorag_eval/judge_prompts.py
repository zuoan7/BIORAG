"""BIORAG Eval v2 — Qwen judge prompts with 4-point scale (1.0/0.75/0.5/0.0/null).

Changes from v1:
- 4-point scale: 1.0=excellent, 0.75=acceptable, 0.5=partial, 0.0=fail, null=not_applicable
- Route-aware metric dispatch (only applicable metrics per category)
- Answer_completeness for summary, comparison_axis_covered for comparison
- Less strict rubric: "核心事实正确但细节略少" → 0.75
"""
from __future__ import annotations
from typing import Any

MAX_SUPPORT_CHARS = 2500


def _format_support(support: list[dict[str, str]]) -> str:
    parts = []
    total = 0
    for i, item in enumerate(support, 1):
        text = item.get("text", "")[:600]
        doc_id = item.get("doc_id", "")[:20]
        line = f"[{i}] [{doc_id}] {text}"
        if total + len(line) > MAX_SUPPORT_CHARS:
            break
        parts.append(line)
        total += len(line)
    return "\n".join(parts) if parts else "(no support evidence)"


def _format_answer_key(record: dict[str, Any]) -> str:
    parts = []
    if record.get("answer_key", "").strip():
        parts.append(f"Answer Key: {record['answer_key'][:500]}")
    if record.get("expected_behavior", "").strip():
        parts.append(f"Expected Behavior: {record['expected_behavior'][:500]}")
    if record.get("reference", "").strip():
        parts.append(f"Reference: {record['reference'][:500]}")
    return "\n".join(parts) if parts else "(no reference available)"


_QWEN_INSTRUCTION = (
    "Reply with a single JSON object. No markdown fences, no extra text. "
    'Keep "rationale" under 80 characters.'
)

_SCALE_4PT = """Score scale:
- 1.0: Excellent — all key points correct and well-supported.
- 0.75: Acceptable — core content correct, minor omissions or imprecise wording.
- 0.5: Partial — some correct content, but notable gaps or weak support.
- 0.0: Fail — incorrect, unsupported, or missing critical information.
- null: Not applicable (no reference to compare against)."""


# ── 1. faithfulness_v3 ──────────────────────────────────────────────────
def faithfulness_prompt(record: dict[str, Any]) -> str:
    support = _format_support(record.get("selected_support", []))
    return f"""You are a biomedical QA evaluator. Judge ONLY whether the claims ALREADY MADE in the answer are supported by the evidence.

Question (for context): {record['question'][:500]}

Evidence:
{support}

Answer:
{record['answer'][:1200]}

CRITICAL: You are ONLY judging the claims the answer actually makes. Do NOT penalize the answer for what it does NOT say. Incompleteness, conciseness, and missing detail are NOT faithfulness issues — they are evaluated separately.

Score the faithfulness of the answer's stated claims:
- 1.0: All key factual claims in the answer are directly supported by the evidence.
- 0.75: Core factual claims are supported; minor imprecise wording but no unsupported factual assertions.
- 0.5: Some factual claims are supported, but one or more important claims lack clear evidence support.
- 0.0: The answer's main factual conclusions are unsupported by the evidence, or contain clear hallucination.

Return JSON:
{{"faithfulness": 1.0|0.75|0.5|0.0, "unsupported_claims": ["brief"], "rationale": "<=80 chars"}}

{_QWEN_INSTRUCTION}"""


# ── 2. answer_relevance_v3 ─────────────────────────────────────────────
def answer_relevance_prompt(record: dict[str, Any]) -> str:
    return f"""You are a biomedical QA evaluator. Judge ONLY whether the answer responds to the question topic. Do NOT evaluate correctness, completeness, evidence support, or conciseness.

Question: {record['question'][:500]}

Answer:
{record['answer'][:1200]}

CRITICAL: You are ONLY judging topic relevance. Accuracy, completeness, evidence quality, and conciseness are evaluated separately. An inaccurate or incomplete answer that addresses the question topic can still be highly relevant.

Score the answer's relevance to the question:
- 1.0: Answer directly responds to the question's core topic.
- 0.75: Answer addresses the question topic but may be somewhat general or include minor digressions.
- 0.5: Answer is partially related to the question but only addresses part of it or is loosely connected.
- 0.0: Answer is largely off-topic or does not respond to the question.

Return JSON:
{{"answer_relevance": 1.0|0.75|0.5|0.0, "off_topic": true|false, "rationale": "<=80 chars"}}

{_QWEN_INSTRUCTION}"""


# ── 3. evidence_recall_v2 ──────────────────────────────────────────────
def evidence_recall_prompt(record: dict[str, Any]) -> str:
    support = _format_support(record.get("selected_support", []))
    answer_key = _format_answer_key(record)
    has_ref = bool((record.get("reference") or "").strip() or (record.get("answer_key") or "").strip() or (record.get("expected_behavior") or "").strip())
    if not has_ref:
        return '{"evidence_recall": null, "missing_evidence": [], "rationale": "no reference/answer_key"}'
    return f"""You are a biomedical QA evaluator. Judge whether the evidence covers the information needed for the question.

Question: {record['question'][:500]}

Evidence:
{support}

{answer_key}

{_SCALE_4PT}

For evidence_recall: does the evidence cover the key information required?
- 1.0: Evidence covers all key information needed to answer the question.
- 0.75: Evidence covers most key information, minor gaps acceptable.
- 0.5: Evidence covers some information, but significant gaps remain.
- 0.0: Evidence lacks critical information needed for the question.
- null: No reference to judge against.

Return JSON:
{{"evidence_recall": 1.0|0.75|0.5|0.0|null, "missing_evidence": ["brief"], "rationale": "<=80 chars"}}

{_QWEN_INSTRUCTION}"""


# ── 4. answer_accuracy_v2 ──────────────────────────────────────────────
def answer_accuracy_prompt(record: dict[str, Any]) -> str:
    answer_key = _format_answer_key(record)
    has_ref = bool((record.get("reference") or "").strip() or (record.get("answer_key") or "").strip())
    if not has_ref:
        return '{"answer_accuracy": null, "missing_key_points": [], "wrong_claims": [], "rationale": "no reference available"}'
    return f"""You are a biomedical QA evaluator. Compare the answer against the expected answer/reference.

Question: {record['question'][:500]}

Expected Answer / Reference:
{answer_key}

Actual Answer:
{record['answer'][:1200]}

{_SCALE_4PT}

For answer_accuracy:
- 1.0: Answer matches expected content, no factual errors.
- 0.75: Answer mostly correct, minor omissions or one small inaccuracy.
- 0.5: Partially correct, some omissions or minor factual errors.
- 0.0: Major factual errors or missing key points.
- null: No reference to compare against.

Return JSON:
{{"answer_accuracy": 1.0|0.75|0.5|0.0|null, "missing_key_points": ["brief"], "wrong_claims": ["brief"], "rationale": "<=80 chars"}}

{_QWEN_INSTRUCTION}"""


# ── 5. answer_completeness_v2 (for summary/review) ─────────────────────
def answer_completeness_prompt(record: dict[str, Any]) -> str:
    support = _format_support(record.get("selected_support", []))
    answer_key = _format_answer_key(record)
    return f"""You are a biomedical QA evaluator for summary/completeness questions.

Question: {record['question'][:500]}

Evidence:
{support}

{answer_key}

Answer:
{record['answer'][:1200]}

{_SCALE_4PT}

For answer_completeness: does the answer cover the key points from the evidence?
- 1.0: Fully covers all key points from the evidence.
- 0.75: Covers most key points, minor omissions.
- 0.5: Covers some key points, notable gaps.
- 0.0: Misses most key points or too brief.

Return JSON:
{{"answer_completeness": 1.0|0.75|0.5|0.0, "missing_key_points": ["brief"], "rationale": "<=80 chars"}}

{_QWEN_INSTRUCTION}"""


# ── 6. negative_abstention_v2 ──────────────────────────────────────────
def negative_abstention_prompt(record: dict[str, Any]) -> str:
    support = _format_support(record.get("selected_support", []))
    return f"""You are a biomedical QA evaluator for negative/non-answer questions.

Question (should receive an abstention or no-answer): {record['question'][:400]}

Evidence:
{support}

Answer:
{record['answer'][:800]}

Expected Behavior: {record.get('expected_behavior', 'abstain')[:200]}
Citation Count: {record.get('citation_count', 0)}

{_SCALE_4PT}

For abstention_correctness:
- 1.0: Clearly refuses to answer or states evidence is insufficient. No unsupported conclusions.
- 0.75: Uses hedging/caveat language while providing limited information without unsupported claims.
- 0.5: Expresses uncertainty but still provides weakly-supported statements.
- 0.0: Gives a definite unsupported conclusion or makes up facts.

Return JSON:
{{"abstention_correctness": 1.0|0.75|0.5|0.0, "unsupported_answer_given": true|false, "rationale": "<=80 chars"}}

{_QWEN_INSTRUCTION}"""


# ── 7. comparison_v2 ───────────────────────────────────────────────────
def comparison_prompt(record: dict[str, Any]) -> str:
    support = _format_support(record.get("selected_support", []))
    return f"""You are a biomedical QA evaluator for comparison questions.

Question: {record['question'][:500]}

Evidence:
{support}

Answer:
{record['answer'][:1200]}

Expected Behavior: {record.get('expected_behavior', '')[:300]}

{_SCALE_4PT}

Evaluate:
- both_branches_covered: Does evidence/answer cover BOTH sides?
- comparison_axis_covered: Is the comparison criterion addressed?
- comparison_faithfulness: Are claims in the comparison supported by evidence? (Note: this is comparison-specific, separate from global faithfulness.)

Return JSON:
{{"both_branches_covered": 1.0|0.75|0.5|0.0, "comparison_axis_covered": 1.0|0.75|0.5|0.0, "comparison_faithfulness": 1.0|0.75|0.5|0.0, "rationale": "<=80 chars"}}

{_QWEN_INSTRUCTION}"""


# ── 8. numeric_v2 ──────────────────────────────────────────────────────
def numeric_prompt(record: dict[str, Any]) -> str:
    support = _format_support(record.get("selected_support", []))
    answer_key = _format_answer_key(record)
    return f"""You are a biomedical QA evaluator for numeric questions.

Question: {record['question'][:500]}

Evidence:
{support}

{answer_key}

Answer:
{record['answer'][:1200]}

{_SCALE_4PT}

Evaluate numeric_accuracy and unit_correctness against evidence/expected answer.
- unit_correct: true if units match, false if wrong, null if no units involved.

Return JSON:
{{"numeric_accuracy": 1.0|0.75|0.5|0.0|null, "unit_correct": true|false|null, "wrong_numbers": ["brief"], "rationale": "<=80 chars"}}

{_QWEN_INSTRUCTION}"""


# ── Prompt dispatch v2 ─────────────────────────────────────────────────
PROMPT_VERSIONS = {
    "faithfulness": "v3.0",
    "answer_relevance": "v3.0",
    "evidence_recall": "v2.0",
    "answer_accuracy": "v2.0",
    "answer_completeness": "v2.0",
    "abstention_correctness": "v2.0",
    "both_branches_covered": "v3.1",
    "comparison_axis_covered": "v3.1",
    "comparison_faithfulness": "v3.1",
    "numeric_accuracy": "v2.0",
    "unit_correct": "v2.0",
}

PROMPT_BUILDERS = {
    "faithfulness": faithfulness_prompt,
    "answer_relevance": answer_relevance_prompt,
    "evidence_recall": evidence_recall_prompt,
    "answer_accuracy": answer_accuracy_prompt,
    "answer_completeness": answer_completeness_prompt,
    "abstention_correctness": negative_abstention_prompt,
    "both_branches_covered": comparison_prompt,
    "comparison_axis_covered": comparison_prompt,
    "comparison_faithfulness": comparison_prompt,
    "numeric_accuracy": numeric_prompt,
    "unit_correct": numeric_prompt,
}
