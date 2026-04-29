#!/usr/bin/env python3
"""Validate enterprise_ragas_eval_v1.json dataset integrity and consistency."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "eval" / "datasets" / "enterprise_ragas_eval_v1.json"

REQUIRED_FIELDS = [
    "id", "question", "reference", "tags", "scenario", "ability",
    "difficulty", "risk_level", "expected_behavior",
    "doc_ids", "sections", "source_files",
    "expected_doc_ids", "expected_source_files", "expected_sections",
    "expected_route", "expected_min_citations", "notes",
]

OPTIONAL_FIELDS = [
    "expected_section_groups", "related_doc_ids", "related_source_files",
    "accepted_doc_ids", "accepted_source_files", "accepted_routes",
    "comparison_branches", "expected_min_doc_coverage",
    "allow_partial_if_doc_coverage", "allow_partial_answer",
    "expected_answer_mode",
]


def validate() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Parse JSON
    try:
        with open(DATASET_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"FATAL: Cannot parse JSON: {e}")
        return 1

    if not isinstance(data, list):
        print("FATAL: Dataset must be a JSON array")
        return 1

    print(f"Dataset: {DATASET_PATH}")
    print(f"Sample count: {len(data)}")

    # 2. Sample count
    if len(data) != 100:
        errors.append(f"Expected 100 samples, got {len(data)}")

    # 3. IDs continuous, no duplicates
    ids = [rec.get("id", "") for rec in data]
    id_set = set(ids)
    if len(id_set) != len(ids):
        dupes = [i for i, c in Counter(ids).items() if c > 1]
        errors.append(f"Duplicate sample IDs: {dupes}")
    expected_ids = {f"ent_{n:03d}" for n in range(1, 101)}
    missing = expected_ids - id_set
    extra = id_set - expected_ids
    if missing:
        errors.append(f"Missing IDs: {sorted(missing)}")
    if extra:
        errors.append(f"Unexpected IDs: {sorted(extra)}")

    # 4. Required fields
    for rec in data:
        sid = rec.get("id", "?")
        for field in REQUIRED_FIELDS:
            if field not in rec:
                errors.append(f"{sid}: missing required field '{field}'")

    # 5. expected_section_groups
    has_groups = sum(1 for rec in data if rec.get("expected_section_groups") is not None)
    if has_groups < len(data):
        warnings.append(f"Only {has_groups}/{len(data)} samples have expected_section_groups")

    # 6. doc_ids / source_files consistency
    for rec in data:
        sid = rec.get("id", "?")
        docs = rec.get("expected_doc_ids") or []
        sources = rec.get("expected_source_files") or []
        if len(docs) != len(sources):
            errors.append(f"{sid}: expected_doc_ids({len(docs)}) != expected_source_files({len(sources)})")
        for d, s in zip(docs, sources):
            if d.replace("doc_", "") not in s:
                warnings.append(f"{sid}: doc {d} and source {s} may not correspond")

    # 7. related_doc_ids / related_source_files consistency
    for rec in data:
        sid = rec.get("id", "?")
        rel_docs = rec.get("related_doc_ids") or []
        rel_srcs = rec.get("related_source_files") or []
        if len(rel_docs) != len(rel_srcs):
            errors.append(f"{sid}: related_doc_ids({len(rel_docs)}) != related_source_files({len(rel_srcs)})")

    # 8. related_doc_ids must not overlap with expected_doc_ids
    for rec in data:
        sid = rec.get("id", "?")
        rel_docs = set(rec.get("related_doc_ids") or [])
        exp_docs = set(rec.get("expected_doc_ids") or [])
        overlap = rel_docs & exp_docs
        if overlap:
            errors.append(f"{sid}: related_doc_ids overlap with expected_doc_ids: {overlap}")

    # 9. negative_case / abstain samples
    for rec in data:
        sid = rec.get("id", "?")
        tags = rec.get("tags") or []
        behavior = rec.get("expected_behavior", "")
        is_abstain = "negative_case" in tags or "abstain" in tags or behavior == "abstain_when_insufficient"
        if is_abstain:
            if rec.get("expected_min_citations", 0) != 0:
                warnings.append(f"{sid}: abstain sample should have expected_min_citations=0")
        else:
            if rec.get("expected_min_citations", 1) == 0 and (rec.get("expected_doc_ids") or []):
                warnings.append(f"{sid}: non-negative sample has expected_min_citations=0 but expected_doc_ids present")

    # 10. Statistics
    route_dist = Counter(rec.get("expected_route", "?") for rec in data)
    scenario_dist = Counter(rec.get("scenario", "?") for rec in data)
    behavior_dist = Counter(rec.get("expected_behavior", "?") for rec in data)

    section_dist = Counter()
    group_dist = Counter()
    for rec in data:
        for s in (rec.get("expected_sections") or []):
            section_dist[s] += 1
        for g in (rec.get("expected_section_groups") or []):
            group_dist[g] += 1

    ft_count = sum(1 for rec in data if "Full Text" in (rec.get("expected_sections") or []))
    ba_count = sum(1 for rec in data if "BODY_ANY" in (rec.get("expected_section_groups") or []))
    rel_count = sum(1 for rec in data if rec.get("related_doc_ids"))

    print(f"\n--- Statistics ---")
    print(f"Route distribution: {dict(route_dist)}")
    print(f"Scenario distribution: {dict(scenario_dist)}")
    print(f"Behavior distribution: {dict(behavior_dist)}")
    print(f"Full Text samples: {ft_count}")
    print(f"BODY_ANY samples: {ba_count}")
    print(f"Samples with related_doc_ids: {rel_count}")
    print(f"Expected sections: {dict(section_dist.most_common(10))}")
    print(f"Expected section groups: {dict(group_dist.most_common())}")

    print(f"\n--- Validation ---")
    print(f"Errors: {len(errors)}")
    for e in errors:
        print(f"  ERROR: {e}")
    print(f"Warnings: {len(warnings)}")
    for w in warnings:
        print(f"  WARN: {w}")

    if errors:
        print("\nVALIDATION FAILED")
        return 1
    print("\nVALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(validate())
