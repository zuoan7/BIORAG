#!/usr/bin/env python3
"""
Generate per-sample diagnostics ledger from existing smoke100 reports.
Reads raw_records from old_baseline.json and v2_stable.json,
loads chunks.jsonl for indexed section lookup,
and outputs eval_metric_diagnostics_ledger.json + .md.

Usage:
  conda activate bge
  python scripts/evaluation/build_diagnostics_ledger.py \
    reports/evaluation/ad_hoc/generation_smoke100/20260428_220340
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag.domain.config import Settings  # noqa: E402

# ── helpers ──────────────────────────────────────────────────────────────────


def _ns(v: Any) -> str:
    return str(v or "").strip()


def _sorted_set(items: list[str]) -> list[str]:
    return sorted({_ns(i) for i in items if _ns(i)})


def _norm(s: str) -> str:
    return _ns(s).lower()


# ── section normalizer ────────────────────────────────────────────────────────


class SectionNormalizer:
    """Semantic section grouping: canonical form + semantic group.

    matching rules:
      - expected "Abstract"            → only actual "abstract"
      - expected "Full Text" / "body"  → any actual with semantic_group=body
      - expected "Results"             → actual "results" or "results_discussion"
      - expected "Discussion"          → actual "discussion" or "results_discussion"
      - expected "Methods"             → actual "methods" (incl. M&M, Experimental Section, etc.)
      - expected "Introduction"        → actual "intro" (incl. Background)
      - expected "Conclusion"          → actual "conclusion" or "conclusions"
    """

    _CANONICAL: dict[str, tuple[tuple[str, ...], str]] = {
        "abstract":           (("abstract",), "abstract"),
        "title":              (("title",), "title"),
        "intro":              (("introduction", "background"), "body"),
        "methods":            (("methods", "materials and methods", "experimental section",
                                "experimental procedures", "m&m", "methods and materials",
                                "method", "methods and materials"), "body"),
        "results":            (("results",), "body"),
        "discussion":         (("discussion",), "body"),
        "results_discussion": (("results and discussion", "r&d", "results & discussion",
                                "discussion and results"), "body"),
        "conclusion":         (("conclusion", "conclusions"), "body"),
        "body":               (("full text", "body", "general body", "fulltext"), "body"),
        "unknown":            (("unknown",), "unknown"),
    }

    _ALIAS_TO_CANONICAL: dict[str, str] = {}
    _CANONICAL_TO_GROUP: dict[str, str] = {}

    @classmethod
    def _init(cls) -> None:
        if cls._ALIAS_TO_CANONICAL:
            return
        for canon, (aliases, group) in cls._CANONICAL.items():
            cls._CANONICAL_TO_GROUP[canon] = group
            for alias in aliases:
                cls._ALIAS_TO_CANONICAL[alias] = canon

    @classmethod
    def canonical(cls, section: str) -> str:
        cls._init()
        return cls._ALIAS_TO_CANONICAL.get(_norm(section), "unknown")

    @classmethod
    def semantic_group(cls, section: str) -> str:
        cls._init()
        return cls._CANONICAL_TO_GROUP.get(cls.canonical(section), "unknown")

    @classmethod
    def expected_matches_actual(cls, expected: str, actual: str) -> bool:
        """Semantic match: does expected section name match actual section name?"""
        exp_canon = cls.canonical(expected)
        act_canon = cls.canonical(actual)
        act_group = cls.semantic_group(actual)

        if exp_canon == act_canon:
            return True
        # "Full Text" matches any body-section actual
        if exp_canon == "body" and act_group == "body":
            return True
        # "Results" ↔ "Results and Discussion" (both directions)
        if exp_canon == "results" and act_canon == "results_discussion":
            return True
        if exp_canon == "discussion" and act_canon == "results_discussion":
            return True
        if exp_canon == "results_discussion" and act_canon in ("results", "discussion"):
            return True
        # "Introduction" ↔ "Background"
        if exp_canon == "intro" and act_canon == "intro":
            return True
        # "Methods" ↔ all method variants
        if exp_canon == "methods" and act_canon == "methods":
            return True
        # "Conclusion" ↔ "Conclusions"
        if exp_canon == "conclusion" and act_canon == "conclusion":
            return True
        return False

    @classmethod
    def any_expected_matches_any_actual(cls, expected_list: list[str], actual_list: list[str]) -> bool:
        return any(
            cls.expected_matches_actual(exp, act)
            for exp in expected_list
            for act in actual_list
        )

    @classmethod
    def is_body_section(cls, section: str) -> bool:
        return cls.semantic_group(section) == "body"

    @classmethod
    def is_full_text_expected(cls, expected_list: list[str]) -> bool:
        return any(cls.canonical(s) == "body" for s in expected_list)


# ── chunk index ──────────────────────────────────────────────────────────────


def load_chunk_index() -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    """Return (doc_id -> ordered_sections, doc_id -> set of unique sections)."""
    settings = Settings.from_env()
    path = Path(settings.kb.chunk_jsonl)
    if not path.exists():
        print(f"[warn] chunk_jsonl not found: {path}", file=sys.stderr)
        return {}, {}

    doc_sections: dict[str, list[str]] = defaultdict(list)
    doc_section_set: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            doc_id = _ns(item.get("doc_id"))
            section = _ns(item.get("section"))
            if not doc_id or not section:
                continue
            doc_sections[doc_id].append(section)
            doc_section_set[doc_id].add(section)
    return {d: list(s) for d, s in doc_sections.items()}, {d: s for d, s in doc_section_set.items()}


# ── section label diagnostics ────────────────────────────────────────────────


def check_section_label_issue(
    expected_sections: list[str],
    indexed_sections: set[str],
) -> dict[str, Any]:
    """Diagnose expected vs indexed section label mismatches using semantic normalizer."""
    if not expected_sections:
        return {"issue": False, "reason": "", "fuzzy_flags": [], "near_matches": [], "indexed_sections": sorted(indexed_sections)}

    SN = SectionNormalizer
    indexed_lower = {s.lower() for s in indexed_sections}
    fuzzy_flags: list[str] = []
    near_matches: list[str] = []
    semantic_matches: list[str] = []

    for exp in expected_sections:
        exp_canon = SN.canonical(exp)
        exp_lower = exp.lower()

        # Check semantic matches against index
        for idx_s in indexed_sections:
            if SN.expected_matches_actual(exp, idx_s):
                semantic_matches.append(f"'{exp}' (canon={exp_canon}) matches index '{idx_s}' (canon={SN.canonical(idx_s)})")
                break

        # Flag still-unresolved issues
        if exp_canon == "body":
            has_body_in_index = any(SN.is_body_section(s) for s in indexed_sections)
            if not has_body_in_index:
                fuzzy_flags.append(f"expected 'Full Text' but no body-section in index")
            elif not semantic_matches:
                fuzzy_flags.append(f"expected 'Full Text' — indexed body sections exist but not matched")
        elif exp_lower in ("results", "discussion"):
            if "results and discussion" in indexed_lower:
                fuzzy_flags.append(f"expected '{exp}' but index uses 'Results and Discussion' (merged)")
        elif exp_lower == "results and discussion":
            if "results" in indexed_lower and "results and discussion" not in indexed_lower:
                fuzzy_flags.append(f"expected 'Results and Discussion' but index uses 'Results' only")
        elif exp_lower in ("materials and methods", "methods and materials"):
            if "methods" in indexed_lower and exp_lower not in indexed_lower:
                fuzzy_flags.append(f"expected '{exp}' but index uses 'Methods'")

        if not semantic_matches and not fuzzy_flags:
            for idx_s in indexed_sections:
                idx_lower = idx_s.lower()
                if exp_lower in idx_lower or idx_lower in exp_lower:
                    near_matches.append(f"'{exp}' ≈ '{idx_s}'")
                    break

    issue = bool(fuzzy_flags) or (not any(SN.expected_matches_actual(s, idx) for s in expected_sections for idx in indexed_sections) and bool(near_matches))
    reason = "; ".join(fuzzy_flags) if fuzzy_flags else ("near matches found" if near_matches else "")
    return {
        "issue": issue,
        "reason": reason,
        "fuzzy_flags": fuzzy_flags,
        "near_matches": near_matches,
        "semantic_matches": semantic_matches,
        "indexed_sections": sorted(indexed_sections),
    }
    return {
        "issue": issue,
        "reason": reason,
        "fuzzy_flags": fuzzy_flags,
        "near_matches": near_matches,
        "indexed_sections": sorted(indexed_sections),
    }


# ── per-sample ledger ────────────────────────────────────────────────────────


def build_sample_ledger(
    raw: dict[str, Any],
    meta: dict[str, Any],
    doc_section_set: dict[str, set[str]],
) -> dict[str, Any]:
    sample_id = raw.get("id", "")
    question = raw.get("question", "")
    gv2 = (raw.get("debug") or {}).get("generation_v2") or {}
    candidates = gv2.get("candidates") or []
    support_pack = raw.get("support_pack") or []
    citations = raw.get("citations") or []
    answer_mode = raw.get("answer_mode", "")
    route = raw.get("route", "")
    expected_route = meta.get("expected_route", "")

    expected_doc_ids = [d for d in (meta.get("expected_doc_ids") or []) if _ns(d)]
    accepted_doc_ids = [d for d in (meta.get("accepted_doc_ids") or []) if _ns(d)]
    eligible_docs = _sorted_set(expected_doc_ids + accepted_doc_ids)
    expected_sections_raw = [s for s in (meta.get("expected_sections") or []) if _ns(s)]

    candidate_docs = _sorted_set([c.get("doc_id", "") for c in candidates])
    support_docs = _sorted_set([s.get("doc_id", "") for s in support_pack])
    citation_docs = _sorted_set([c.get("doc_id", "") for c in citations])

    candidate_doc_set = set(candidate_docs)
    support_doc_set = set(support_docs)
    citation_doc_set = set(citation_docs)
    eligible_set = set(eligible_docs)

    expected_docs_in_retrieval = sorted(eligible_set & candidate_doc_set)
    expected_docs_in_citations = sorted(eligible_set & citation_doc_set)
    missing_from_retrieval = sorted(eligible_set - candidate_doc_set)
    missing_from_citations = sorted(eligible_set - citation_doc_set)

    # Doc failure category
    if eligible_set:
        if not expected_docs_in_retrieval:
            doc_failure = "retrieval_doc_miss"
        elif not expected_docs_in_citations:
            mode_lower = _norm(answer_mode)
            if "full" in mode_lower:
                doc_failure = "retrieved_but_not_cited_full_answer"
            else:
                doc_failure = "retrieved_but_not_cited_partial_or_refuse"
        else:
            doc_failure = "citation_doc_hit"
    else:
        doc_failure = "no_expected_doc"

    # Section diagnostics
    retrieved_sections = _sorted_set([c.get("section", "") for c in candidates])
    support_sections = _sorted_set([s.get("section", "") for s in support_pack])
    citation_sections = _sorted_set([c.get("section", "") for c in citations])

    retrieved_section_set = set(retrieved_sections)
    citation_section_set = set(citation_sections)
    expected_section_set = set(expected_sections_raw)

    # ── strict (unchanged) ──
    strict_section_hit = bool(expected_section_set & citation_section_set) if expected_section_set else None
    strict_retrieval_hit = bool(expected_section_set & retrieved_section_set) if expected_section_set else None

    # ── case-insensitive (old normalized) ──
    expected_lower = {s.lower() for s in expected_sections_raw}
    citation_lower = {s.lower() for s in citation_sections}
    case_insensitive_hit = bool(expected_lower & citation_lower) if expected_section_set else None

    # ── semantic normalized (new) ──
    SN = SectionNormalizer
    normalized_section_hit = SN.any_expected_matches_any_actual(expected_sections_raw, citation_sections) if expected_section_set else None
    retrieval_section_hit = SN.any_expected_matches_any_actual(expected_sections_raw, retrieved_sections) if expected_section_set else None

    # normalized hit details: which expected matched which actual
    normalized_hit_pairs: list[dict[str, str]] = []
    if expected_section_set and citation_sections:
        for exp in expected_sections_raw:
            for act in citation_sections:
                if SN.expected_matches_actual(exp, act):
                    normalized_hit_pairs.append({"expected": exp, "actual": act, "expected_canonical": SN.canonical(exp), "actual_canonical": SN.canonical(act)})

    # Full Text special diagnostics
    has_full_text_expected = SN.is_full_text_expected(expected_sections_raw)
    full_text_normalized_hit = False
    if has_full_text_expected and citation_sections:
        full_text_normalized_hit = any(
            SN.canonical(exp) == "body" and SN.is_body_section(act)
            for exp in expected_sections_raw
            for act in citation_sections
        )

    # indexed sections per expected doc (use normalizer)
    indexed_sections_by_expected_doc: dict[str, dict[str, Any]] = {}
    index_possible = False
    index_possible_normalized = False
    for doc_id in eligible_docs:
        idx_sections = doc_section_set.get(doc_id, set())
        label_diag = check_section_label_issue(expected_sections_raw, idx_sections)
        indexed_sections_by_expected_doc[doc_id] = label_diag
        if idx_sections:
            # strict index possible
            idx_lower = {s.lower() for s in idx_sections}
            if expected_lower & idx_lower:
                index_possible = True
            # normalized index possible
            if SN.any_expected_matches_any_actual(expected_sections_raw, sorted(idx_sections)):
                index_possible_normalized = True

    index_possible_section_hit = index_possible if expected_section_set else None
    index_possible_normalized_hit = index_possible_normalized if expected_section_set else None

    # ── section failure category ──
    if expected_section_set:
        if normalized_section_hit:
            section_failure = "citation_section_hit"
        elif retrieval_section_hit:
            section_failure = "retrieved_but_not_cited_section"
        elif index_possible_normalized:
            section_failure = "retrieval_section_miss"
        elif has_full_text_expected:
            # Full Text is expected but no body-section was retrieved
            section_failure = "retrieval_section_miss"
        elif any(
            (indexed_sections_by_expected_doc.get(d) or {}).get("fuzzy_flags")
            for d in eligible_docs
        ):
            section_failure = "eval_label_mismatch"
        elif any(
            (indexed_sections_by_expected_doc.get(d) or {}).get("indexed_sections")
            for d in eligible_docs
        ):
            section_failure = "index_missing_expected_section"
        else:
            section_failure = "unknown"
    else:
        section_failure = "no_expected_section"

    # Identify section collapse in index
    index_collapsed: list[str] = []
    for doc_id in eligible_docs:
        idx_set = doc_section_set.get(doc_id, set())
        idx_lower = {s.lower() for s in idx_set}
        for exp in expected_sections_raw:
            exp_lower = exp.lower()
            if exp_lower in ("results and discussion",):
                if "results" in idx_lower and exp_lower not in idx_lower:
                    index_collapsed.append(f"'{exp}' → index has separate Results/Discussion for {doc_id}")
            elif exp_lower in ("results", "discussion"):
                if "results and discussion" in idx_lower:
                    index_collapsed.append(f"'{exp}' merged into 'Results and Discussion' for {doc_id}")

    expected_sections_normalized = [f"{s} → {SectionNormalizer.canonical(s)}" for s in expected_sections_raw]

    return {
        # ── id / question / route ──
        "sample_id": sample_id,
        "question": question,
        "expected_route": _ns(expected_route),
        "actual_route": _ns(route),
        "answer_type": answer_mode,

        # ── doc ──
        "expected_doc_ids": eligible_docs,
        "retrieved_doc_ids": candidate_docs,
        "support_pack_doc_ids": support_docs,
        "citation_doc_ids": citation_docs,
        "expected_docs_in_retrieval": expected_docs_in_retrieval,
        "expected_docs_in_citations": expected_docs_in_citations,
        "missing_expected_docs_from_retrieval": missing_from_retrieval,
        "missing_expected_docs_from_citations": missing_from_citations,
        "doc_failure_category": doc_failure,

        # ── section ──
        "expected_sections_raw": expected_sections_raw,
        "expected_sections_normalized": expected_sections_normalized,
        "indexed_sections_by_expected_doc": indexed_sections_by_expected_doc,
        "retrieved_sections": retrieved_sections,
        "support_pack_sections": support_sections,
        "citation_sections": citation_sections,
        "strict_section_hit": strict_section_hit,
        "case_insensitive_section_hit": case_insensitive_hit,
        "normalized_section_hit": normalized_section_hit,
        "index_possible_section_hit": index_possible_section_hit,
        "index_possible_normalized_hit": index_possible_normalized_hit,
        "retrieval_section_hit": retrieval_section_hit,
        "strict_retrieval_hit": strict_retrieval_hit,
        "citation_section_hit": normalized_section_hit,
        "has_full_text_expected": has_full_text_expected,
        "full_text_normalized_hit": full_text_normalized_hit,
        "normalized_hit_pairs": normalized_hit_pairs,
        "section_failure_category": section_failure,
        "index_section_collapse": index_collapsed if index_collapsed else None,
    }


# ── aggregate ────────────────────────────────────────────────────────────────


def compute_aggregate(ledger_entries: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(ledger_entries)
    if total == 0:
        return {}

    # doc metrics
    retrieval_doc_hit = 0
    citation_doc_hit = 0
    doc_total = 0
    retrieved_not_cited = 0
    not_retrieved = 0
    full_miss = 0
    partial_miss = 0
    refuse_miss = 0

    # section metrics
    # strict
    strict_hit = 0
    strict_retrieval_hit = 0
    # case-insensitive
    case_insensitive_hit = 0
    # normalized (semantic)
    norm_hit = 0
    index_possible_hit = 0
    index_possible_norm_hit = 0
    retrieval_sec_hit = 0
    citation_sec_hit = 0
    # Full Text
    full_text_samples = 0
    full_text_norm_hit = 0
    sec_total = 0

    failure_category_counts: dict[str, int] = Counter()
    section_failure_counts: dict[str, int] = Counter()
    route_mismatch_count = 0

    for e in ledger_entries:
        # doc
        has_expected = bool(e["expected_doc_ids"])
        if has_expected:
            doc_total += 1
            if e["expected_docs_in_retrieval"]:
                retrieval_doc_hit += 1
            if e["expected_docs_in_citations"]:
                citation_doc_hit += 1

        fc = e["doc_failure_category"]
        failure_category_counts[fc] += 1

        if fc == "retrieved_but_not_cited_full_answer":
            full_miss += 1
        elif fc == "retrieved_but_not_cited_partial_or_refuse":
            if "partial" in _norm(e["answer_type"]):
                partial_miss += 1
            elif "refuse" in _norm(e["answer_type"]):
                refuse_miss += 1
        elif fc == "retrieval_doc_miss" or fc == "not_retrieved":
            not_retrieved += 1

        if fc == "retrieved_but_not_cited_full_answer" or fc == "retrieved_but_not_cited_partial_or_refuse":
            retrieved_not_cited += 1

        if e["expected_route"] and e["actual_route"] and _norm(e["expected_route"]) != _norm(e["actual_route"]):
            route_mismatch_count += 1

        # section
        has_sec = bool(e["expected_sections_raw"])
        if has_sec:
            sec_total += 1
            if e["strict_section_hit"]:
                strict_hit += 1
            if e["strict_retrieval_hit"]:
                strict_retrieval_hit += 1
            if e["case_insensitive_section_hit"]:
                case_insensitive_hit += 1
            if e["normalized_section_hit"]:
                norm_hit += 1
            if e["index_possible_section_hit"]:
                index_possible_hit += 1
            if e["index_possible_normalized_hit"]:
                index_possible_norm_hit += 1
            if e["retrieval_section_hit"]:
                retrieval_sec_hit += 1
            if e["citation_section_hit"]:
                citation_sec_hit += 1
            if e.get("has_full_text_expected"):
                full_text_samples += 1
                if e.get("full_text_normalized_hit"):
                    full_text_norm_hit += 1

        section_failure_counts[e["section_failure_category"]] += 1

    def _rate(hits: int, tot: int) -> float:
        return round(hits / tot, 4) if tot > 0 else float("nan")

    # strict_miss_but_normalized_hit
    strict_miss_norm_hit = sum(
        1 for e in ledger_entries
        if e["expected_sections_raw"] and not e["strict_section_hit"] and e["normalized_section_hit"]
    )

    return {
        "sample_count": total,
        "doc": {
            "retrieval_doc_hit_rate": _rate(retrieval_doc_hit, doc_total),
            "citation_doc_hit_rate": _rate(citation_doc_hit, doc_total),
            "retrieval_doc_hit_count": retrieval_doc_hit,
            "citation_doc_hit_count": citation_doc_hit,
            "doc_total_with_expected": doc_total,
            "retrieved_but_not_cited_count": retrieved_not_cited,
            "not_retrieved_count": not_retrieved,
            "full_answer_missing_expected_doc_count": full_miss,
            "partial_answer_missing_expected_doc_count": partial_miss,
            "refuse_missing_expected_doc_count": refuse_miss,
        },
        "section": {
            "strict_section_hit_rate": _rate(strict_hit, sec_total),
            "case_insensitive_section_hit_rate": _rate(case_insensitive_hit, sec_total),
            "normalized_section_hit_rate": _rate(norm_hit, sec_total),
            "index_possible_section_hit_rate": _rate(index_possible_hit, sec_total),
            "index_possible_normalized_hit_rate": _rate(index_possible_norm_hit, sec_total),
            "retrieval_section_hit_rate": _rate(retrieval_sec_hit, sec_total),
            "citation_section_hit_rate": _rate(citation_sec_hit, sec_total),
            "strict_hit_count": strict_hit,
            "case_insensitive_hit_count": case_insensitive_hit,
            "norm_hit_count": norm_hit,
            "index_possible_count": index_possible_hit,
            "index_possible_normalized_count": index_possible_norm_hit,
            "retrieval_hit_count": retrieval_sec_hit,
            "citation_hit_count": citation_sec_hit,
            "section_total_with_expected": sec_total,
            "strict_miss_but_normalized_hit": strict_miss_norm_hit,
            "full_text_expected_samples": full_text_samples,
            "full_text_normalized_hit_samples": full_text_norm_hit,
        },
        "failure": {
            "route_mismatch_count": route_mismatch_count,
            "doc_failure_distribution": dict(sorted(failure_category_counts.items())),
            "section_failure_distribution": dict(sorted(section_failure_counts.items())),
        },
    }


# ── markdown ─────────────────────────────────────────────────────────────────


def build_markdown(ledger: list[dict[str, Any]], aggregate: dict[str, Any], group_label: str) -> str:
    agg = aggregate
    doc = agg.get("doc", {})
    sec = agg.get("section", {})
    fail = agg.get("failure", {})

    lines = [
        f"# Diagnostics Ledger: {group_label}",
        "",
        f"Sample count: {agg.get('sample_count', 0)}",
        "",
        "## Aggregate Metrics",
        "",
        "### Doc",
        "",
        "| metric | value |",
        "|---|---|",
        f"| retrieval_doc_hit_rate | `{doc.get('retrieval_doc_hit_rate')}` |",
        f"| citation_doc_hit_rate | `{doc.get('citation_doc_hit_rate')}` |",
        f"| retrieval_doc_hit_count | `{doc.get('retrieval_doc_hit_count')}` / `{doc.get('doc_total_with_expected')}` |",
        f"| citation_doc_hit_count | `{doc.get('citation_doc_hit_count')}` / `{doc.get('doc_total_with_expected')}` |",
        f"| retrieved_but_not_cited_count | `{doc.get('retrieved_but_not_cited_count')}` |",
        f"| not_retrieved_count | `{doc.get('not_retrieved_count')}` |",
        f"| full_answer_missing_expected_doc | `{doc.get('full_answer_missing_expected_doc_count')}` |",
        f"| partial_answer_missing_expected_doc | `{doc.get('partial_answer_missing_expected_doc_count')}` |",
        f"| refuse_missing_expected_doc | `{doc.get('refuse_missing_expected_doc_count')}` |",
        "",
        "### Section",
        "",
        "#### Before vs After: Normalized Section Matching",
        "",
        "| metric | before (case-insensitive) | after (semantic) |",
        "|---|---|---|",
        f"| normalized_section_hit_rate | `{sec.get('case_insensitive_section_hit_rate')}` | `{sec.get('normalized_section_hit_rate')}` |",
        f"| index_possible_section_hit_rate | `{sec.get('index_possible_section_hit_rate')}` | `{sec.get('index_possible_normalized_hit_rate')}` |",
        f"| citation_hit_count | `{sec.get('case_insensitive_hit_count')}` / `{sec.get('section_total_with_expected')}` | `{sec.get('norm_hit_count')}` / `{sec.get('section_total_with_expected')}` |",
        "",
        f"strict_miss_but_normalized_hit: `{sec.get('strict_miss_but_normalized_hit')}` samples",
        f"full_text_expected_samples: `{sec.get('full_text_expected_samples')}`",
        f"full_text_normalized_hit_samples: `{sec.get('full_text_normalized_hit_samples')}`",
        "",
        "#### Full Pipeline Metrics (semantic)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| strict_section_hit_rate | `{sec.get('strict_section_hit_rate')}` |",
        f"| normalized_section_hit_rate | `{sec.get('normalized_section_hit_rate')}` |",
        f"| index_possible_section_hit_rate | `{sec.get('index_possible_section_hit_rate')}` |",
        f"| index_possible_normalized_hit_rate | `{sec.get('index_possible_normalized_hit_rate')}` |",
        f"| retrieval_section_hit_rate | `{sec.get('retrieval_section_hit_rate')}` |",
        f"| citation_section_hit_rate | `{sec.get('citation_section_hit_rate')}` |",
        "",
        "#### Remaining Gaps",
        "",
        f"- strict_hit: `{sec.get('strict_hit_count')}` / case_insensitive: `{sec.get('case_insensitive_hit_count')}` / norm_hit: `{sec.get('norm_hit_count')}` / index_possible: `{sec.get('index_possible_count')}` / index_possible_norm: `{sec.get('index_possible_normalized_count')}` / retrieval_hit: `{sec.get('retrieval_hit_count')}` / `{sec.get('section_total_with_expected')}`",
        f"- strict_miss_but_normalized_hit: `{sec.get('strict_miss_but_normalized_hit')}`",
        "",
        "### Failure Distribution",
        "",
        f"- route_mismatch_count: `{fail.get('route_mismatch_count')}`",
        f"- doc_failure: `{json.dumps(fail.get('doc_failure_distribution', {}), ensure_ascii=False)}`",
        f"- section_failure: `{json.dumps(fail.get('section_failure_distribution', {}), ensure_ascii=False)}`",
        "",
        "## Per-Sample Ledger",
        "",
    ]

    # Doc section summary table
    lines += [
        "### Doc Status",
        "",
        "| id | question | route | answer | expected_docs | in_retrieval | in_citations | missing_retrieval | missing_citations | doc_failure |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in ledger:
        q = e["question"][:60]
        lines.append(
            f"| `{e['sample_id']}` | {q} | `{e['actual_route']}` | `{e['answer_type']}` | "
            f"`{e['expected_doc_ids']}` | `{e['expected_docs_in_retrieval']}` | "
            f"`{e['expected_docs_in_citations']}` | `{e['missing_expected_docs_from_retrieval']}` | "
            f"`{e['missing_expected_docs_from_citations']}` | `{e['doc_failure_category']}` |"
        )

    # Section summary table
    lines += [
        "",
        "### Section Status",
        "",
        "| id | expected_sections | retrieved_sections | citation_sections | strict | norm | idx_possible | retrieval | citation | sec_failure |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in ledger:
        lines.append(
            f"| `{e['sample_id']}` | `{e['expected_sections_raw']}` | "
            f"`{e['retrieved_sections']}` | `{e['citation_sections']}` | "
            f"`{e['strict_section_hit']}` | `{e['normalized_section_hit']}` | "
            f"`{e['index_possible_section_hit']}` | `{e['retrieval_section_hit']}` | "
            f"`{e['citation_section_hit']}` | `{e['section_failure_category']}` |"
        )

    # Section label issue samples
    label_issues = [e for e in ledger if any(
        d.get("issue") for d in (e.get("indexed_sections_by_expected_doc") or {}).values()
    )]
    if label_issues:
        lines += [
            "",
            "### Section Label Issues",
            "",
        ]
        for e in label_issues:
            for doc_id, diag in (e.get("indexed_sections_by_expected_doc") or {}).items():
                if diag.get("issue"):
                    lines.append(
                        f"- `{e['sample_id']}` doc=`{doc_id}`: {diag.get('reason')} "
                        f"(fuzzy: {diag.get('fuzzy_flags')}, near: {diag.get('near_matches')})"
                    )

    # Full Text diagnostic samples
    full_text_samples_list = [e for e in ledger if e.get("has_full_text_expected")]
    if full_text_samples_list:
        ft_hit = [e for e in full_text_samples_list if e.get("full_text_normalized_hit")]
        ft_miss = [e for e in full_text_samples_list if not e.get("full_text_normalized_hit")]
        lines += [
            "",
            f"### Full Text Expected Samples ({len(full_text_samples_list)} total)",
            "",
            f"- Full Text normalized hit: `{len(ft_hit)}` / `{len(full_text_samples_list)}`",
            f"- Full Text normalized miss: `{len(ft_miss)}` / `{len(full_text_samples_list)}`",
            "",
        ]
        if ft_hit:
            lines.append("#### Full Text Normalized Hit")
            lines.append("")
            for e in ft_hit:
                lines.append(f"- `{e['sample_id']}`: citations={e['citation_sections']}, pairs={e.get('normalized_hit_pairs')}")
            lines.append("")
        if ft_miss:
            lines.append("#### Full Text Normalized Miss (no body-section in citations)")
            lines.append("")
            for e in ft_miss:
                lines.append(f"- `{e['sample_id']}`: retrieved={e['retrieved_sections']}, cited={e['citation_sections']}, answer={e['answer_type']}")
            lines.append("")

    # Strict miss but normalized hit
    strict_miss_norm = [e for e in ledger if e["expected_sections_raw"] and not e["strict_section_hit"] and e["normalized_section_hit"]]
    if strict_miss_norm:
        lines += [
            f"### Strict Miss But Normalized Hit ({len(strict_miss_norm)} samples)",
            "",
        ]
        for e in strict_miss_norm:
            lines.append(f"- `{e['sample_id']}`: expected={e['expected_sections_raw']}, cited={e['citation_sections']}, pairs={e.get('normalized_hit_pairs')}")
        lines.append("")

    # Remaining eval_label_mismatch
    remaining_elm = [e for e in ledger if e["section_failure_category"] == "eval_label_mismatch"]
    if remaining_elm:
        lines += [
            f"### Remaining eval_label_mismatch ({len(remaining_elm)} samples)",
            "",
        ]
        for e in remaining_elm:
            lines.append(f"- `{e['sample_id']}`: expected={e['expected_sections_raw']}, cited={e['citation_sections']}, retrieved={e['retrieved_sections']}")
        lines.append("")

    # True retrieval_section_miss
    retrieval_sm = [e for e in ledger if e["section_failure_category"] == "retrieval_section_miss"]
    if retrieval_sm:
        lines += [
            f"### Retrieval Section Miss ({len(retrieval_sm)} samples)",
            "",
        ]
        for e in retrieval_sm:
            lines.append(f"- `{e['sample_id']}`: expected={e['expected_sections_raw']}, indexed_possible_norm={e['index_possible_normalized_hit']}, retrieved={e['retrieved_sections']}")
        lines.append("")

    # Retrieved but not cited section
    rbnc = [e for e in ledger if e["section_failure_category"] == "retrieved_but_not_cited_section"]
    if rbnc:
        lines += [
            f"### Retrieved But Not Cited Section ({len(rbnc)} samples)",
            "",
        ]
        for e in rbnc:
            lines.append(f"- `{e['sample_id']}`: expected={e['expected_sections_raw']}, retrieved={e['retrieved_sections']}, cited={e['citation_sections']}")
        lines.append("")

    # Index collapse samples
    collapsed = [e for e in ledger if e.get("index_section_collapse")]
    if collapsed:
        lines += [
            "",
            "### Index Section Collapse",
            "",
        ]
        for e in collapsed:
            for c in e["index_section_collapse"]:
                lines.append(f"- `{e['sample_id']}`: {c}")

    lines.append("")
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python build_diagnostics_ledger.py <report_dir>", file=sys.stderr)
        return 1

    report_dir = Path(sys.argv[1])
    if not report_dir.is_dir():
        print(f"Not a directory: {report_dir}", file=sys.stderr)
        return 1

    print(f"Loading reports from {report_dir} ...")
    old_path = report_dir / "old_baseline.json"
    v2_path = report_dir / "v2_stable.json"

    if not old_path.exists() or not v2_path.exists():
        print("Missing old_baseline.json or v2_stable.json", file=sys.stderr)
        return 1

    old_data = json.loads(old_path.read_text(encoding="utf-8"))
    v2_data = json.loads(v2_path.read_text(encoding="utf-8"))

    print("Loading chunk index ...")
    doc_sections_list, doc_section_set = load_chunk_index()
    print(f"  Loaded sections for {len(doc_section_set)} docs")

    # Also load the original dataset to get dataset_meta per sample
    dataset_path = ROOT / "data/eval/datasets/enterprise_ragas_eval_v1.json"
    print(f"Loading dataset from {dataset_path} ...")
    dataset_items = json.loads(dataset_path.read_text(encoding="utf-8"))
    meta_by_id: dict[str, dict[str, Any]] = {}
    for item in dataset_items:
        mid = _ns(item.get("id"))
        if mid:
            meta_by_id[mid] = {
                "expected_doc_ids": item.get("expected_doc_ids") or item.get("doc_ids") or [],
                "accepted_doc_ids": item.get("accepted_doc_ids") or [],
                "expected_sections": item.get("expected_sections") or [],
                "expected_route": _ns(item.get("expected_route")),
            }
    print(f"  Loaded meta for {len(meta_by_id)} samples")

    for group_key, group_label, group_data in [
        ("old_baseline", "Old Baseline (Legacy Pipeline)", old_data),
        ("v2_stable", "V2 Stable (Extractive Only)", v2_data),
    ]:
        print(f"\nBuilding ledger for {group_key} ...")
        raw_records = group_data.get("raw_records") or []
        if not raw_records:
            print(f"  No raw_records in {group_key}, skipping")
            continue

        ledger_entries: list[dict[str, Any]] = []
        for raw in raw_records:
            sample_id = raw.get("id", "")
            meta = meta_by_id.get(sample_id, {})
            entry = build_sample_ledger(raw, meta, doc_section_set)
            ledger_entries.append(entry)

        aggregate = compute_aggregate(ledger_entries)

        json_out = report_dir / f"eval_metric_diagnostics_ledger_{group_key}.json"
        md_out = report_dir / f"eval_metric_diagnostics_report_{group_key}.md"

        output = {
            "group": group_key,
            "label": group_label,
            "aggregate": aggregate,
            "ledger": ledger_entries,
        }

        json_out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> {json_out} ({len(ledger_entries)} entries)")

        md = build_markdown(ledger_entries, aggregate, group_label)
        md_out.write_text(md, encoding="utf-8")
        print(f"  -> {md_out}")

        # Print key metrics
        doc = aggregate.get("doc", {})
        sec = aggregate.get("section", {})
        fail = aggregate.get("failure", {})
        print(f"  doc: retrieval_hit={doc.get('retrieval_doc_hit_rate')} "
              f"citation_hit={doc.get('citation_doc_hit_rate')} "
              f"retrieved_not_cited={doc.get('retrieved_but_not_cited_count')} "
              f"not_retrieved={doc.get('not_retrieved_count')}")
        print(f"  section: strict={sec.get('strict_section_hit_rate')} "
              f"norm={sec.get('normalized_section_hit_rate')} "
              f"idx_possible={sec.get('index_possible_section_hit_rate')} "
              f"retrieval={sec.get('retrieval_section_hit_rate')} "
              f"citation={sec.get('citation_section_hit_rate')}")
        print(f"  doc_failure: {fail.get('doc_failure_distribution')}")
        print(f"  section_failure: {fail.get('section_failure_distribution')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
