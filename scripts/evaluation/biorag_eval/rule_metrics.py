"""BIORAG Eval v1 — Layer 1: Deterministic rule metrics. No LLM calls."""

from __future__ import annotations
from typing import Any


def doc_recall(expected_ids: list[str], retrieved_ids: list[str]) -> float:
    """Fraction of expected doc_ids present in retrieved doc_ids."""
    if not expected_ids:
        return 1.0  # No expected → perfect recall (or flag differently?)
    expected_set = set(str(d) for d in expected_ids)
    retrieved_set = set(str(d) for d in retrieved_ids)
    if not expected_set:
        return 1.0
    return len(expected_set & retrieved_set) / len(expected_set)


def doc_recall_support(record: dict[str, Any]) -> float:
    """Fraction of expected_doc_ids in selected_support_doc_ids."""
    return doc_recall(record.get("expected_doc_ids", []), record.get("selected_support_doc_ids", []))


def doc_recall_citation(record: dict[str, Any]) -> float:
    """Fraction of expected_doc_ids in cited_doc_ids."""
    return doc_recall(record.get("expected_doc_ids", []), record.get("cited_doc_ids", []))


def expected_doc_in_support(record: dict[str, Any]) -> bool:
    """At least one expected doc in selected_support."""
    return doc_recall_support(record) > 0


def expected_doc_cited(record: dict[str, Any]) -> bool:
    """At least one expected doc in citations."""
    return doc_recall_citation(record) > 0


def route_match(record: dict[str, Any]) -> bool:
    """Predicted route matches expected route."""
    exp = (record.get("expected_route") or "").lower().strip()
    pred = (record.get("route_pred") or "").lower().strip()
    if not exp or not pred:
        return False
    # Normalize: factoid/factoid_precision → factoid
    norm_exp = exp.replace("_precision", "").replace("_review", "")
    norm_pred = pred.replace("_precision", "").replace("_review", "")
    if norm_exp == "comparison" and "comparison" in norm_pred:
        return True
    if norm_exp == "summary" and "summary" in norm_pred:
        return True
    if norm_exp == "negative" and "negative" in norm_pred:
        return True
    return norm_exp == norm_pred


def wrong_doc_citation(record: dict[str, Any]) -> bool:
    """Any cited doc is not in expected or accepted doc list."""
    expected = set(str(d) for d in (record.get("expected_doc_ids") or []))
    cited = set(str(d) for d in (record.get("cited_doc_ids") or []))
    if not cited:
        return False
    if not expected:
        return False  # Cannot judge without expected docs
    return not cited.issubset(expected)


def negative_citation_zero(record: dict[str, Any]) -> bool:
    """Negative samples must have zero citations."""
    if not record.get("is_negative", False):
        return True  # Not applicable for non-negative
    return record.get("citation_count", 0) == 0


def citation_inflation(record: dict[str, Any], expected_min: int = 3) -> bool:
    """Citation count far exceeds expected_min with no extra doc coverage."""
    return record.get("citation_count", 0) > expected_min * 3


def smoke_alignment_status(record: dict[str, Any]) -> str:
    """Return smoke alignment: pass, fail, or unknown."""
    if record.get("smoke_real_P0") == "yes":
        return "fail"
    if record.get("smoke_real_P0") == "no":
        return "pass"
    return "unknown"


def compute_all_rule_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """Compute all rule metrics for an EvalRecord."""
    return {
        "sample_id": record["sample_id"],
        "category": record.get("category", ""),
        "expected_route": record.get("expected_route", ""),
        "route_pred": record.get("route_pred", ""),
        "route_match": route_match(record),
        "doc_recall_support": round(doc_recall_support(record), 4),
        "doc_recall_citation": round(doc_recall_citation(record), 4),
        "expected_doc_in_support": expected_doc_in_support(record),
        "expected_doc_cited": expected_doc_cited(record),
        "citation_count": record.get("citation_count", 0),
        "wrong_doc_citation": wrong_doc_citation(record),
        "negative_citation_zero": negative_citation_zero(record),
        "smoke_real_P0": record.get("smoke_real_P0", ""),
        "smoke_failure_class": record.get("smoke_failure_class", ""),
        "smoke_alignment": smoke_alignment_status(record),
        "notes": "",
    }
