"""BIORAG Eval v1 schemas. Lightweight dataclass-style dict schema."""

from __future__ import annotations
from typing import Any


# ── EvalRecord: per-sample evaluation record ───────────────────────────
def make_eval_record(
    sample_id: str,
    split: str = "",
    category: str = "",
    expected_route: str = "",
    question: str = "",
    answer: str = "",
    selected_support: list[dict[str, str]] | None = None,
    cited_doc_ids: list[str] | None = None,
    selected_support_doc_ids: list[str] | None = None,
    expected_doc_ids: list[str] | None = None,
    expected_source_files: list[str] | None = None,
    expected_behavior: str = "",
    answer_key: str = "",
    reference: str = "",
    is_negative: bool = False,
    route_pred: str = "",
    citation_count: int = 0,
    smoke_real_P0: str = "no",
    smoke_failure_class: str = "",
    smoke_first_loss_stage: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a unified EvalRecord dict."""
    return {
        "sample_id": sample_id,
        "split": split,
        "category": category,
        "expected_route": expected_route,
        "question": question,
        "answer": answer,
        "selected_support": selected_support or [],
        "cited_doc_ids": cited_doc_ids or [],
        "selected_support_doc_ids": selected_support_doc_ids or [],
        "expected_doc_ids": expected_doc_ids or [],
        "expected_source_files": expected_source_files or [],
        "expected_behavior": expected_behavior,
        "answer_key": answer_key,
        "reference": reference,
        "is_negative": is_negative,
        "route_pred": route_pred,
        "citation_count": citation_count,
        "smoke_real_P0": smoke_real_P0,
        "smoke_failure_class": smoke_failure_class,
        "smoke_first_loss_stage": smoke_first_loss_stage,
    }


# ── Rule metric result ──────────────────────────────────────────────────
RULE_METRIC_SCHEMA: dict[str, Any] = {
    "route_match": "bool — predicted route matches expected_route",
    "doc_recall_final": "float — fraction of expected_doc_ids in final_chunks",
    "doc_recall_support": "float — fraction of expected_doc_ids in selected_support",
    "doc_recall_citation": "float — fraction of expected_doc_ids in citations",
    "expected_doc_in_final": "bool — at least one expected_doc in final_chunks",
    "expected_doc_in_selected_support": "bool — at least one expected_doc in selected_support",
    "expected_doc_cited": "bool — at least one expected_doc cited",
    "citation_count": "int — number of citations",
    "wrong_doc_citation": "bool — any cited doc not in expected/accepted",
    "citation_inflation": "bool — citation_count > expected_min_citations by large margin",
    "negative_citation_zero": "bool — negative samples have citation_count=0",
    "first_loss_stage": "str — first pipeline stage where expected_doc was lost",
}


# ── Qwen judge metric schemas ──────────────────────────────────────────
FAITHFULNESS_SCHEMA = {
    "faithfulness": "1.0 | 0.5 | 0.0",
    "unsupported_claims": "list[str]",
    "rationale": "str, <=80 chars",
}

EVIDENCE_RECALL_SCHEMA = {
    "evidence_recall": "1.0 | 0.5 | 0.0 | null",
    "missing_evidence": "list[str]",
    "rationale": "str, <=80 chars",
}

ANSWER_ACCURACY_SCHEMA = {
    "answer_accuracy": "1.0 | 0.5 | 0.0 | null",
    "missing_key_points": "list[str]",
    "wrong_claims": "list[str]",
    "rationale": "str, <=80 chars",
}

NEGATIVE_ABSTENTION_SCHEMA = {
    "abstention_correctness": "1.0 | 0.5 | 0.0",
    "unsupported_answer_given": "bool",
    "rationale": "str, <=80 chars",
}

COMPARISON_SCHEMA = {
    "both_branches_covered": "1.0 | 0.5 | 0.0",
    "comparison_axis_covered": "1.0 | 0.5 | 0.0",
    "faithfulness": "1.0 | 0.5 | 0.0",
    "rationale": "str, <=80 chars",
}

NUMERIC_SCHEMA = {
    "numeric_accuracy": "1.0 | 0.5 | 0.0 | null",
    "unit_correct": "bool | null",
    "wrong_numbers": "list[str]",
    "rationale": "str, <=80 chars",
}

ANSWER_RELEVANCE_SCHEMA = {
    "answer_relevance": "1.0 | 0.5 | 0.0",
    "irrelevant_parts": "list[str]",
    "rationale": "str, <=80 chars",
}


# ── Scoring scale v2 ────────────────────────────────────────────────────
SCORE_SCALE = [1.0, 0.75, 0.5, 0.0]
SCORE_NULL = None
PASS_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.5


def score_bucket(score: float | None) -> str:
    """Map score to bucket: pass, partial, fail, not_applicable."""
    if score is None:
        return "not_applicable"
    if score >= 0.75:
        return "pass"
    if score >= 0.5:
        return "partial"
    return "fail"


# ── Metric applicability policy v2 (route-aware) ────────────────────────
# Each route/category gets a specific set of applicable judge metrics.
# This avoids computing abstention_correctness on non-negative,
# comparison_quality on factoid, etc.

_ROUTE_METRICS: dict[str, list[str]] = {
    "factoid": ["faithfulness", "answer_relevance", "evidence_recall", "answer_accuracy"],
    "factoid_precision": ["faithfulness", "answer_relevance", "evidence_recall", "answer_accuracy"],
    "summary": ["faithfulness", "answer_relevance", "evidence_recall", "answer_completeness", "answer_accuracy"],
    "summary_review": ["faithfulness", "answer_relevance", "evidence_recall", "answer_completeness", "answer_accuracy"],
    "comparison": ["faithfulness", "both_branches_covered", "comparison_axis_covered", "comparison_faithfulness", "answer_relevance", "answer_accuracy"],
    "multi_doc_comparison": ["faithfulness", "both_branches_covered", "comparison_axis_covered", "comparison_faithfulness", "answer_relevance", "answer_accuracy"],
    "cross_lingual": ["faithfulness", "answer_relevance", "evidence_recall", "answer_accuracy"],
    "method_result_numeric": ["faithfulness", "numeric_accuracy", "unit_correct", "evidence_recall", "answer_accuracy"],
    "table_figure_caption": ["faithfulness", "numeric_accuracy", "unit_correct", "evidence_recall", "answer_accuracy"],
    "negative_trigger_robustness": ["abstention_correctness"],
}


def _normalize_category(cat: str) -> str:
    cat_l = (cat or "").lower()
    for key in _ROUTE_METRICS:
        if key in cat_l or cat_l in key:
            return key
    return "factoid"  # default


def applicable_metrics(record: dict[str, Any]) -> list[str]:
    """Return list of applicable judge metric names for this record."""
    if record.get("is_negative", False):
        return ["abstention_correctness"]

    cat = _normalize_category(record.get("category", "factoid"))
    metrics = list(_ROUTE_METRICS.get(cat, _ROUTE_METRICS["factoid"]))

    # Remove answer_accuracy if no reference or answer_key
    has_ref = bool((record.get("reference") or "").strip() or (record.get("answer_key") or "").strip())
    if not has_ref and "answer_accuracy" in metrics:
        metrics.remove("answer_accuracy")
    # Remove evidence_recall if no expected_behavior/answer_key
    has_behavior = bool((record.get("expected_behavior") or "").strip())
    if not has_ref and not has_behavior and "evidence_recall" in metrics:
        metrics.remove("evidence_recall")

    return metrics


def metric_applicable(record: dict[str, Any], metric_name: str) -> tuple[bool, str]:
    """Return (applicable, skip_reason). v2: route-aware dispatch."""
    allowed = applicable_metrics(record)
    if metric_name in allowed:
        return True, ""
    return False, f"not_applicable_for_{record.get('category', 'unknown')}"
