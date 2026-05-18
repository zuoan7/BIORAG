#!/usr/bin/env python3
"""Quality check script for Phase 12G holdout50 repair_v2."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_PATH = ROOT / "data/evaluation/phase12g_parent_holdout50.jsonl"
CHUNKS_PATH = ROOT / "data/paper_round1/chunks/chunks.jsonl"
SMOKE100_PATH = ROOT / "data/eval/datasets/enterprise_ragas_smoke100.json"

EXPECTED_CATS = {
    "summary": 10,
    "comparison": 10,
    "table_figure_caption": 10,
    "method_result_numeric": 5,
    "negative_trigger_robustness": 5,
    "factoid": 10,
}
ALLOWED_RETRIEVAL_MODES = {"open_retrieval", "targeted_retrieval"}
TOPIC_TERMS = [
    "2′-FL",
    "2'-FL",
    "2-FL",
    "3-FL",
    "6′-SL",
    "6'-SL",
    "3′-SL",
    "3'-SL",
    "Neu5Ac",
    "D-allulose",
    "PHA",
    "HAC1",
    "UPR",
    "Kex2",
    "Keio",
    "OPN",
    "FAM20",
    "FAM20A",
    "FAM20C",
    "NanR",
    "NanT",
    "GH1",
    "CRISPR-TMSD",
    "PASTE",
    "IFN",
    "IFN-a2b",
    "IFN-α2b",
    "Psd1",
    "glutarate",
    "戊二酸",
    "sugarcane molasses",
    "糖蜜",
    "bidirectional promoter",
    "双向启动子",
    "polycistronic",
    "多顺反子",
    "MoSpc2",
    "F1SBP",
    "expression cassette",
    "expression vector",
]
LEGACY_HIGH_RISK_IDS = {
    "h50_sum_003",
    "h50_sum_006",
    "h50_sum_007",
    "h50_sum_008",
    "h50_tf_003",
    "h50_tf_004",
    "h50_tf_008",
    "h50_tf_009",
    "h50_fact_002",
    "h50_fact_005",
    "h50_fact_008",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_dataset_items(path: Path) -> list[dict]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    data = load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "items", "records", "samples", "questions"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    return []


def normalize_question(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"doc_\d+", "", text)
    text = text.replace("figure", "fig").replace("table", "tab")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[，。！？、；：\"'（）【】《》…—\-,.!?;:()\[\]{}_/]", "", text)
    return text


def extract_topics(text: str) -> set[str]:
    low = text.lower()
    hits = {term for term in TOPIC_TERMS if term.lower() in low}
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-αβγ′/]{2,}", text):
        if token.lower().startswith("doc_"):
            continue
        if len(token) >= 4 and any(ch.isalpha() for ch in token):
            hits.add(token)
    return hits


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def main() -> int:
    if not HOLDOUT_PATH.exists():
        print("FAIL: holdout file not found")
        return 1

    samples = load_jsonl(HOLDOUT_PATH)
    chunks = load_jsonl(CHUNKS_PATH)
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks if chunk.get("chunk_id")}
    all_doc_ids = {chunk.get("doc_id") for chunk in chunks if chunk.get("doc_id")}

    issues: list[str] = []
    warnings: list[str] = []

    if len(samples) != 50:
        issues.append(f"Expected 50 samples, got {len(samples)}")

    ids = [sample.get("sample_id") for sample in samples]
    dup_ids = [sid for sid, count in Counter(ids).items() if count > 1]
    if dup_ids:
        issues.append(f"Duplicate sample_ids: {dup_ids}")

    cats = Counter(sample.get("category") for sample in samples)
    for category, expected in EXPECTED_CATS.items():
        if cats.get(category, 0) != expected:
            issues.append(f"Category {category}: expected {expected}, got {cats.get(category, 0)}")

    retrieval_counts = Counter(sample.get("retrieval_mode") for sample in samples)
    for mode in retrieval_counts:
        if mode not in ALLOWED_RETRIEVAL_MODES:
            issues.append(f"Unsupported retrieval_mode: {mode}")
    if retrieval_counts.get("open_retrieval", 0) < 35:
        issues.append(f"open_retrieval count too low: {retrieval_counts.get('open_retrieval', 0)}")
    if retrieval_counts.get("targeted_retrieval", 0) > 15:
        issues.append(f"targeted_retrieval count too high: {retrieval_counts.get('targeted_retrieval', 0)}")

    open_doc_leaks: list[str] = []
    missing_behavior: list[str] = []
    needs_review: list[str] = []
    unique_docs: set[str] = set()

    for sample in samples:
        sid = sample["sample_id"]
        question = sample.get("question", "")
        mode = sample.get("retrieval_mode")
        expected_behavior = sample.get("expected_behavior")

        if not question.strip():
            issues.append(f"{sid}: empty question")
        if not isinstance(expected_behavior, list) or not expected_behavior:
            issues.append(f"{sid}: expected_behavior missing or empty")
            missing_behavior.append(sid)

        if not sample.get("expected_doc_ids") and not sample.get("expected_source_files"):
            issues.append(f"{sid}: expected_doc_ids and expected_source_files are both empty")

        for doc_id in sample.get("expected_doc_ids", []):
            unique_docs.add(doc_id)
            if doc_id not in all_doc_ids:
                issues.append(f"{sid}: expected_doc_id {doc_id} not found in chunks")

        chunk_ids = sample.get("construction_source_chunk_ids", [])
        if not chunk_ids:
            issues.append(f"{sid}: construction_source_chunk_ids empty")
        for chunk_id in chunk_ids:
            if chunk_id not in chunk_by_id:
                issues.append(f"{sid}: chunk {chunk_id} not found in chunks")

        if mode == "open_retrieval":
            if re.search(r"doc_\d+", question):
                issues.append(f"{sid}: open_retrieval question leaks doc_id")
                open_doc_leaks.append(sid)
            if sample.get("doc_ids") or sample.get("source_files") or sample.get("sections"):
                issues.append(f"{sid}: open_retrieval must keep doc_ids/source_files/sections empty")
        elif mode == "targeted_retrieval":
            if not sample.get("doc_ids") and not sample.get("source_files"):
                warnings.append(f"{sid}: targeted_retrieval without doc_ids/source_files")

        if "NEEDS_REVIEW" in str(sample.get("notes", "")):
            needs_review.append(sid)

    smoke100_items = load_dataset_items(SMOKE100_PATH)
    existing_paths = [
        ROOT / "data/eval/datasets/enterprise_ragas_smoke100.json",
        ROOT / "data/eval/datasets/enterprise_ragas_smoke20.json",
        ROOT / "data/eval/datasets/enterprise_ragas_smoke5.json",
        ROOT / "data/eval/datasets/enterprise_ragas_abstention10.json",
        ROOT / "data/eval/datasets/enterprise_ragas_docmiss13.json",
        ROOT / "data/eval/datasets/enterprise_ragas_targeted20.json",
        ROOT / "data/eval/datasets/phase2_precheck.json",
        ROOT / "data/eval/datasets/ragas_eval_sample.jsonl",
        ROOT / "data/eval/datasets/targeted_systemic_fix_subset.json",
        ROOT / "data/eval/round8_comparison_core_regression.json",
        ROOT / "data/eval/round8_comparison_partial_branch.json",
        ROOT / "data/eval/round8_comparison_regression.json",
        ROOT / "data/eval/round8_comparison_retrieval_stress.json",
        ROOT / "data/eval/round8_false_abstention_regression.json",
        ROOT / "data/eval/round8_false_abstention_single_doc_core.json",
        ROOT / "data/eval/round8_multidoc_partial_comparison.json",
        ROOT / "data/eval/round8_retrieval_failure_guardrail.json",
        ROOT / "data/eval/round8_route_ambiguity.json",
        ROOT / "data/evaluation/phase12d_small_smoke_questions.jsonl",
        ROOT / "data/evaluation/phase12e_comparison_regression_questions.jsonl",
        ROOT / "data/evaluation/phase12e_table_figure_diagnostic_questions.jsonl",
        ROOT / "data/evaluation/v5_phase3b_chunk_retrieval_smoke_cases.jsonl",
        ROOT / "data/small_exp/table_retrieval_queries.jsonl",
    ]
    existing_items: list[dict] = []
    for path in existing_paths:
        existing_items.extend(load_dataset_items(path))

    existing_questions = []
    for item in existing_items:
        question = item.get("question") or item.get("query") or ""
        if question:
            existing_questions.append(question)
    existing_norm = {normalize_question(question) for question in existing_questions if question}

    exact_dups: list[str] = []
    near_dups: list[str] = []
    for sample in samples:
        norm = normalize_question(sample["question"])
        if norm in existing_norm:
            exact_dups.append(sample["sample_id"])
            continue
        for existing in existing_norm:
            if len(norm) < 24 or len(existing) < 24:
                continue
            if similarity(norm, existing) >= 0.86:
                near_dups.append(sample["sample_id"])
                break
    if exact_dups:
        issues.append(f"Exact normalized duplicates with existing datasets: {exact_dups}")
    if near_dups:
        issues.append(f"High-similarity normalized duplicates with existing datasets: {near_dups}")

    smoke_norm = []
    for item in smoke100_items:
        question = item.get("question") or item.get("query") or ""
        reference = item.get("reference") or ""
        notes = item.get("notes") or ""
        doc_ids = set(item.get("expected_doc_ids") or item.get("doc_ids") or [])
        smoke_norm.append(
            {
                "id": item.get("id"),
                "norm": normalize_question(question),
                "text": " ".join(x for x in [question, reference, notes] if x),
                "topics": extract_topics(" ".join(x for x in [question, reference, notes] if x)),
                "doc_ids": doc_ids,
            }
        )

    remaining_overlap_rows: list[tuple[str, str]] = []
    remaining_high_risk_ids: list[str] = []
    for sample in samples:
        sid = sample["sample_id"]
        sample_text = " ".join(
            str(sample.get(key, "")) for key in ("question", "notes")
        )
        sample_topics = extract_topics(sample_text)
        sample_docs = set(sample.get("expected_doc_ids", []))
        for smoke in smoke_norm:
            shared_docs = sample_docs & smoke["doc_ids"]
            shared_topics = {topic for topic in sample_topics & smoke["topics"] if len(topic) >= 4}
            sim = similarity(normalize_question(sample["question"]), smoke["norm"])
            if shared_docs and (shared_topics or sim >= 0.72):
                reason_bits = []
                if shared_docs:
                    reason_bits.append(f"shared_doc={sorted(shared_docs)}")
                if shared_topics:
                    reason_bits.append(f"shared_topics={sorted(shared_topics)[:5]}")
                if sim >= 0.72:
                    reason_bits.append(f"sim={sim:.2f}")
                reason = "; ".join(reason_bits)
                remaining_overlap_rows.append((sid, f"{smoke['id']}: {reason}"))
                if sid in LEGACY_HIGH_RISK_IDS:
                    remaining_high_risk_ids.append(sid)
                break

    if len(set(remaining_high_risk_ids)) > 2:
        warnings.append(
            f"Remaining same-doc/same-topic high-risk legacy overlaps: {sorted(set(remaining_high_risk_ids))}"
        )

    tf006 = next(sample for sample in samples if sample["sample_id"] == "h50_tf_006")
    if tf006.get("expected_sections") == ["Title"]:
        issues.append("h50_tf_006: expected_sections cannot stay ['Title'] without explicit justification")

    for sid in ("h50_tf_009", "h50_tf_010"):
        sample = next(sample for sample in samples if sample["sample_id"] == sid)
        chunk_ids = sample.get("construction_source_chunk_ids", [])
        if len(chunk_ids) != len(set(chunk_ids)):
            issues.append(f"{sid}: duplicate construction_source_chunk_ids require repair or explicit consolidation")

    fig_sources = 0
    table_sources = 0
    for sample in samples:
        if sample["category"] != "table_figure_caption":
            continue
        seen = set(sample.get("construction_source_chunk_ids", []))
        for chunk_id in seen:
            chunk = chunk_by_id.get(chunk_id, {})
            if chunk.get("contains_figure_caption"):
                fig_sources += 1
            if chunk.get("contains_table_caption"):
                table_sources += 1
    if fig_sources < 4:
        warnings.append(f"Figure-caption source count is low: {fig_sources}")
    if table_sources < 4:
        warnings.append(f"Table-caption source count is low: {table_sources}")

    verdict = "FAIL" if issues else ("PASS_WITH_WARNINGS" if warnings else "PASS")

    print("# Holdout50 Repair V2 Quality Report")
    print()
    print("## Summary")
    print(f"- File: {HOLDOUT_PATH.relative_to(ROOT)}")
    print(f"- Samples: {len(samples)}")
    print(f"- Unique docs: {len(unique_docs)}")
    print(f"- Verdict: {verdict}")
    print()
    print("## Category Distribution")
    for category in EXPECTED_CATS:
        print(f"- {category}: {cats.get(category, 0)}")
    print()
    print("## Retrieval Mode Distribution")
    print(f"- open_retrieval: {retrieval_counts.get('open_retrieval', 0)}")
    print(f"- targeted_retrieval: {retrieval_counts.get('targeted_retrieval', 0)}")
    print()
    print("## Dedup Summary")
    print(f"- Existing datasets scanned: {len(existing_paths)}")
    print(f"- Exact normalized duplicates vs smoke/existing: {len(exact_dups)}")
    print(f"- High-similarity normalized duplicates vs smoke/existing: {len(near_dups)}")
    print(f"- Remaining same-doc/same-topic smoke100 overlaps: {len(remaining_overlap_rows)}")
    for sid, reason in remaining_overlap_rows:
        print(f"  - {sid}: {reason}")
    print()
    print("## Table/Figure Checks")
    print(f"- Figure-caption source count: {fig_sources}")
    print(f"- Table-caption source count: {table_sources}")
    print()
    print("## Special Sample Checks")
    print(f"- h50_tf_006 expected_sections: {tf006.get('expected_sections')}")
    for sid in ('h50_tf_009', 'h50_tf_010'):
        sample = next(sample for sample in samples if sample['sample_id'] == sid)
        print(f"- {sid} chunks: {sample.get('construction_source_chunk_ids', [])}")
    print()
    print("## NEEDS_REVIEW")
    print(f"- Count: {len(needs_review)}")
    for sid in needs_review:
        print(f"  - {sid}")

    if issues:
        print()
        print(f"## Issues ({len(issues)})")
        for issue in issues:
            print(f"- {issue}")

    if warnings:
        print()
        print(f"## Warnings ({len(warnings)})")
        for warning in warnings:
            print(f"- {warning}")

    print()
    print("## Proceed")
    print(f"- {'Can' if verdict != 'FAIL' else 'Cannot'} proceed to Phase 12G-holdout50-ab")

    return 1 if verdict == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
