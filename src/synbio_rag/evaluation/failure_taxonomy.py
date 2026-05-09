"""Unified evaluation taxonomy — separates raw/real/diagnostic failure classification.

Replaces scattered P0/failure_category logic across evaluation scripts.
Backward-compatible: raw_failure_category and is_raw_p0 preserved alongside corrected fields.
"""
from dataclasses import dataclass, field
from typing import Any


DIAGNOSTIC_FLAGS = [
    "route_mismatch",
    "route_mismatch_false_p0_doc_cited",
    "route_mismatch_true",
    "metric_or_dataset_issue",
    "near_topic_but_expected_doc_miss",
    "translation_drift",
    "implicit_reference_preservation_fail",
    "negative_query_regression",
    "wrong_doc_citation",
    "citation_marker_not_used",
]

REAL_P0_RULES = {
    "doc_miss": "expected doc not in cited_docs AND not in selected_support -> real_P0",
    "zero_citation": "citation_count == 0 -> real_P0",
    "citation_failure": "citation_count < expected_min_citations -> real_P0",
    "wrong_doc_citation": "expected doc not cited, wrong doc cited as primary -> real_P0",
    "negative_query_regression": "negative/abstain query answered with citations when expected refusal -> real_P0",
    "answer_quality_issue": "partial answer, refusal when answer expected, or missing evidence coverage -> real_P0",
    "true_route_regression": "route_mismatch AND (doc not cited OR answer quality issue) -> real_P0",
    "false_route_exclusion": "route_mismatch with doc cited AND no quality issue -> NOT real_P0 (diagnostic only)",
}


@dataclass
class FailureAssessment:
    """Unified failure evaluation for a single sample."""

    sample_id: str = ""
    # Raw (backward-compatible) layer
    raw_failure_category: str = "ok"
    is_raw_p0: bool = False
    raw_p0_reason: str = ""
    # Corrected real layer
    corrected_failure_category: str = "ok"
    is_real_p0: bool = False
    real_p0_reason: str = ""
    # Route mismatch diagnostics
    route_mismatch_type: str = "none"  # false_p0_doc_cited | true_regression | none
    expected_doc_cited: bool = False
    expected_source_file_cited: bool = False
    answer_quality_issue_present: bool = False
    # Classification deltas (set by caller comparing v0/v1)
    should_count_as_real_regression: bool = False
    should_count_as_real_improvement: bool = False
    # Diagnostic flags
    diagnostic_flags: list = field(default_factory=list)
    # Extra context
    notes: str = ""


def _doc_in_set(doc_id: Any, doc_set: Any) -> bool:
    """Check if doc_id appears in a set/list of doc IDs."""
    if isinstance(doc_set, str):
        doc_set = [d.strip() for d in doc_set.split("|") if d.strip()]
    if not isinstance(doc_set, (list, set, tuple)):
        return False
    return str(doc_id) in [str(d) for d in doc_set]


def evaluate_failure(
    raw_failure_category: str,
    doc_hit: bool,
    cited_doc_ids: Any,
    expected_doc_ids: Any,
    expected_source_files: Any = None,
    citation_count: int = 0,
    expected_min_citations: int = 0,
    answer_mode: str = "full",
    is_negative: bool = False,
    route_match: bool = True,
    source_file_hit: bool = False,
) -> FailureAssessment:
    """Unified failure evaluation.

    Args:
        raw_failure_category: The pipeline's raw failure category (ok, doc_miss, route_mismatch, partial_answer).
        doc_hit: Whether expected_doc appears in cited_docs or selected_support.
        cited_doc_ids: List or pipe-separated string of cited doc IDs.
        expected_doc_ids: List or pipe-separated string of expected doc IDs.
        expected_source_files: Optional expected source files for cross-check.
        citation_count: Number of citations in the answer.
        expected_min_citations: Minimum expected citation count.
        answer_mode: The generation's answer mode (full, partial, refuse).
        is_negative: Whether this is a negative/abstain query.
        route_match: Whether actual_route matches expected_route.
        source_file_hit: Whether expected source file was cited.

    Returns:
        FailureAssessment with raw, corrected, and diagnostic fields populated.
    """
    fa = FailureAssessment()

    # ── Normalise inputs ──
    exp_docs = expected_doc_ids
    if isinstance(exp_docs, str):
        exp_docs = [d.strip() for d in exp_docs.split("|") if d.strip()]
    if not isinstance(exp_docs, (list, set, tuple)):
        exp_docs = [exp_docs] if exp_docs else []

    cited = cited_doc_ids
    if isinstance(cited, str):
        cited = [d.strip() for d in cited.split("|") if d.strip()]
    if cited is None:
        cited = []

    fa.expected_doc_cited = any(_doc_in_set(d, cited) for d in exp_docs) if exp_docs else True
    fa.expected_source_file_cited = source_file_hit or fa.expected_doc_cited
    fa.answer_quality_issue_present = answer_mode in ("partial", "refuse")
    fa.sample_id = ""

    # ── Raw layer (backward-compatible) ──
    fa.raw_failure_category = raw_failure_category
    fa.is_raw_p0 = raw_failure_category in ("route_mismatch", "doc_miss") and not is_negative
    fa.raw_p0_reason = raw_failure_category if fa.is_raw_p0 else "ok"

    # ── Corrected real P0 layer ──
    if raw_failure_category == "route_mismatch":
        if fa.expected_doc_cited and not fa.answer_quality_issue_present:
            fa.corrected_failure_category = "route_mismatch_false_p0_doc_cited"
            fa.is_real_p0 = False
            fa.route_mismatch_type = "false_p0_doc_cited"
            fa.real_p0_reason = "false_route_p0_doc_cited"
            fa.diagnostic_flags.append("route_mismatch_false_p0_doc_cited")
        else:
            fa.corrected_failure_category = "route_mismatch_true"
            fa.is_real_p0 = True
            fa.route_mismatch_type = "true_regression"
            fa.real_p0_reason = "true_route_regression"
            fa.diagnostic_flags.append("route_mismatch_true")
        fa.diagnostic_flags.append("route_mismatch")
    elif raw_failure_category == "doc_miss":
        fa.corrected_failure_category = "doc_miss"
        fa.is_real_p0 = True
        fa.real_p0_reason = "doc_miss"
    elif citation_count == 0 and not is_negative:
        fa.corrected_failure_category = "zero_citation"
        fa.is_real_p0 = True
        fa.real_p0_reason = "zero_citation"
    elif not fa.expected_doc_cited and raw_failure_category not in ("ok","partial_answer") and not is_negative:
        fa.corrected_failure_category = "doc_miss_silent"
        fa.is_real_p0 = True
        fa.real_p0_reason = "expected_doc_not_cited"
        fa.diagnostic_flags.append("near_topic_but_expected_doc_miss")
    elif is_negative and citation_count > 0:
        fa.corrected_failure_category = "negative_query_regression"
        fa.is_real_p0 = True
        fa.real_p0_reason = "negative_query_answered"
        fa.diagnostic_flags.append("negative_query_regression")
    else:
        fa.corrected_failure_category = raw_failure_category
        fa.is_real_p0 = False
        fa.real_p0_reason = "ok"

    # Auto-populate diagnostic flags for non-cited expected docs
    if not fa.expected_doc_cited and not fa.is_real_p0:
        fa.diagnostic_flags.append("near_topic_but_expected_doc_miss")

    fa.should_count_as_real_regression = fa.is_real_p0
    return fa
