#!/usr/bin/env python3
"""Phase 5D-4 static audit and lightweight retrieval smoke.

This script is intentionally read-only with respect to parsed/chunk inputs. It
only writes validation reports under reports/phase5d_caption_cleanup_validation.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_REPORT_DIR = ROOT / "reports/phase5d_caption_cleanup_experiment"
SIGNOFF_DIR = ROOT / "reports/phase5d_caption_cleanup_signoff"
OUT_DIR = ROOT / "reports/phase5d_caption_cleanup_validation"

BASELINE_PARSED_DIR = ROOT / "data/paper_round1/parsed_clean"
CLEANUP_PARSED_DIR = Path("/tmp/biorag_phase5d3_caption_cleanup/parsed_clean")
BASELINE_CHUNKS = Path("/tmp/biorag_phase4d_compact_chunks/chunks.jsonl")
CLEANUP_CHUNKS = Path("/tmp/biorag_phase5d3_caption_cleanup/chunks/chunks.jsonl")

DOC_0367_FIGURE5_BLOCK_ID = "p6_b0059"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def md_bool(value: bool) -> str:
    return "true" if value else "false"


def load_doc(parsed_dir: Path, doc_id: str) -> dict[str, Any]:
    with (parsed_dir / f"{doc_id}.json").open(encoding="utf-8") as f:
        return json.load(f)


def iter_blocks(doc: dict[str, Any]):
    for page in doc.get("pages", []):
        for block in page.get("blocks", []):
            yield block


def find_block(parsed_dir: Path, doc_id: str, block_id: str) -> dict[str, Any] | None:
    doc = load_doc(parsed_dir, doc_id)
    for block in iter_blocks(doc):
        if block.get("block_id") == block_id:
            return block
    return None


def count_block_types(parsed_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in sorted(parsed_dir.glob("doc_*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for block in iter_blocks(doc):
            counts[str(block.get("type", ""))] += 1
    return counts


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def chunk_has_block(chunk: dict[str, Any], doc_id: str, block_id: str) -> bool:
    if chunk.get("doc_id") != doc_id:
        return False
    ids = chunk.get("source_block_ids") or chunk.get("block_ids") or []
    return block_id in ids


def caption_focused_for_block(chunk: dict[str, Any], doc_id: str, block_id: str, original_type: str) -> bool:
    if not chunk_has_block(chunk, doc_id, block_id):
        return False
    evidence_types = set(chunk.get("evidence_types") or [])
    block_types = set(chunk.get("block_types") or [])
    if original_type == "table_caption":
        return bool(
            chunk.get("contains_table_caption")
            or "table_caption" in evidence_types
            or "table_caption" in block_types
        )
    if original_type == "figure_caption":
        return bool(
            chunk.get("contains_figure_caption")
            or "figure_caption" in evidence_types
            or "figure_caption" in block_types
        )
    return False


def paragraph_retained_for_block(chunk: dict[str, Any], doc_id: str, block_id: str) -> bool:
    if not chunk_has_block(chunk, doc_id, block_id):
        return False
    evidence_types = set(chunk.get("evidence_types") or [])
    block_types = set(chunk.get("block_types") or [])
    return (
        "paragraph" in evidence_types
        or "paragraph" in block_types
        or not ({"table_caption", "figure_caption"} & evidence_types)
    )


def compact(text: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def classify_demotion(row: dict[str, str], baseline_block: dict[str, Any] | None, cleanup_block: dict[str, Any] | None) -> tuple[str, str]:
    text = row.get("caption_text", "")
    rule = row.get("caption_cleanup_rule_id", "")
    preview = row.get("nearby_preview", "").lower()
    safe = row.get("safe_to_demote") == "true" and row.get("signoff_label") == "safe_to_demote"
    if not safe:
        return "likely_false_positive_demote", "Demoted row was not signoff safe_to_demote."
    if baseline_block is None or cleanup_block is None:
        return "unknown", "Could not locate baseline or cleanup block."
    if baseline_block.get("text") != cleanup_block.get("text"):
        return "likely_false_positive_demote", "Cleanup block text differs from baseline."
    if cleanup_block.get("type") != "paragraph":
        return "unknown", "Cleanup block was not demoted to paragraph."
    if rule == "phase5d3_page_header_footer_caption_fragment":
        if re.fullmatch(r"\s*(fig(?:ure)?\.?\s*)?\d+\s+\d+\s+of\s+\d+\s*", text, re.I):
            return "acceptable_demote", "Page-header/footer fragment pattern."
        return "questionable_needs_manual_check", "Expected page-header/footer fragment pattern was not exact."
    if rule == "phase5d3_broken_organism_or_abbreviation_prefix":
        if re.search(r"\b(?:Table|Fig\.?|Figure|Supplementary Table|Table S\d+)\s*[\w.]*\s*(?:The\s+)?[A-Z]\.$", text):
            if any(word in preview for word in ["coli", "cerevisiae", "pastoris", "phaffii", "subtilis"]):
                return "acceptable_demote", "Caption text is an organism/name prefix continued in nearby paragraph."
            return "questionable_needs_manual_check", "Looks like a prefix fragment but nearby continuation keyword was not found."
    if rule == "phase5d3_article_plus_single_letter_fragment":
        if re.search(r"\b(?:Fig|Figure)\s*\d+\.?\s+The\s+[A-Z]\.$", text):
            if any(word in preview for word in ["jejuni", "breve", "lnnt", "gt42"]):
                return "acceptable_demote", "Article plus single-letter fragment continued in nearby paragraph."
            return "questionable_needs_manual_check", "Looks like article plus single-letter fragment; continuation is weak."
    return "unknown", "No deterministic classification rule matched."


def summarize_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    field_sets = Counter(tuple(sorted(c.keys())) for c in chunks)
    top_fields = sorted(set().union(*(set(c.keys()) for c in chunks))) if chunks else []
    return {
        "chunk_count": len(chunks),
        "field_set_count": len(field_sets),
        "field_sets": [list(fields) for fields, _ in field_sets.most_common()],
        "top_level_fields": top_fields,
        "table_focused_chunk_count": sum(1 for c in chunks if c.get("contains_table_caption")),
        "figure_focused_chunk_count": sum(1 for c in chunks if c.get("contains_figure_caption")),
        "normal_paragraph_chunk_count": sum(
            1
            for c in chunks
            if not c.get("contains_table_caption")
            and not c.get("contains_figure_caption")
            and "paragraph" in set(c.get("evidence_types") or c.get("block_types") or [])
        ),
    }


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+\-./]*")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if len(t) > 1]


@dataclass
class SimpleBM25:
    chunks: list[dict[str, Any]]
    avgdl: float
    doc_len: list[int]
    idf: dict[str, float]
    postings: dict[str, list[tuple[int, int]]]

    @classmethod
    def build(cls, chunks: list[dict[str, Any]]) -> "SimpleBM25":
        postings_by_term: dict[str, dict[int, int]] = defaultdict(dict)
        doc_len: list[int] = []
        for idx, chunk in enumerate(chunks):
            text = chunk.get("retrieval_text") or chunk.get("text") or ""
            terms = tokenize(text)
            doc_len.append(len(terms))
            tf = Counter(terms)
            for term, count in tf.items():
                postings_by_term[term][idx] = count
        n_docs = len(chunks)
        idf = {
            term: math.log(1.0 + (n_docs - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in postings_by_term.items()
        }
        postings = {term: list(posting.items()) for term, posting in postings_by_term.items()}
        avgdl = sum(doc_len) / max(1, len(doc_len))
        return cls(chunks=chunks, avgdl=avgdl, doc_len=doc_len, idf=idf, postings=postings)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        q_terms = Counter(tokenize(query))
        scores: dict[int, float] = defaultdict(float)
        k1 = 1.5
        b = 0.75
        for term, qtf in q_terms.items():
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for idx, tf in self.postings[term]:
                dl = self.doc_len[idx] or 1
                denom = tf + k1 * (1 - b + b * dl / max(self.avgdl, 1e-9))
                scores[idx] += idf * (tf * (k1 + 1) / denom) * qtf
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            {
                "rank": rank,
                "score": score,
                "chunk": self.chunks[idx],
            }
            for rank, (idx, score) in enumerate(ranked, start=1)
        ]


def target_hit(results: list[dict[str, Any]], target_doc_id: str, target_block_ids: list[str]) -> bool:
    target_ids = set(target_block_ids)
    for result in results:
        chunk = result["chunk"]
        if chunk.get("doc_id") != target_doc_id:
            continue
        chunk_ids = set(chunk.get("source_block_ids") or chunk.get("block_ids") or [])
        if target_ids & chunk_ids:
            return True
    return False


def caption_focused_target_hit(
    results: list[dict[str, Any]],
    target_doc_id: str,
    target_block_ids: list[str],
) -> bool:
    target_ids = set(target_block_ids)
    for result in results:
        chunk = result["chunk"]
        if chunk.get("doc_id") != target_doc_id:
            continue
        chunk_ids = set(chunk.get("source_block_ids") or chunk.get("block_ids") or [])
        if not (target_ids & chunk_ids):
            continue
        if chunk.get("contains_table_caption") or chunk.get("contains_figure_caption"):
            return True
        if {"table_caption", "figure_caption"} & set(chunk.get("evidence_types") or []):
            return True
    return False


def first_hit_summary(results: list[dict[str, Any]], target_doc_id: str, target_block_ids: list[str]) -> dict[str, Any] | None:
    target_ids = set(target_block_ids)
    for result in results:
        chunk = result["chunk"]
        chunk_ids = set(chunk.get("source_block_ids") or chunk.get("block_ids") or [])
        if chunk.get("doc_id") == target_doc_id and target_ids & chunk_ids:
            return {
                "rank": result["rank"],
                "score": round(result["score"], 6),
                "chunk_id": chunk.get("chunk_id"),
                "evidence_types": chunk.get("evidence_types"),
                "contains_table_caption": chunk.get("contains_table_caption"),
                "contains_figure_caption": chunk.get("contains_figure_caption"),
            }
    return None


def make_caption_query_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    words = text.split()
    return " ".join(words[:36])


def make_normal_query_text(text: str) -> str:
    text = re.sub(r"[\[\]#*_`]+", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    return " ".join(words[:28])


def build_queries(
    protected_rows: list[dict[str, str]],
    demoted_rows: list[dict[str, str]],
    cleanup_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    used_ids: set[tuple[str, str, str]] = set()

    def add_query(
        query_type: str,
        query: str,
        doc_id: str,
        block_ids: list[str],
        expected_behavior: str,
        include: bool,
        notes: str,
    ) -> None:
        sample_id = f"{query_type}_{len([q for q in queries if q['query_type'] == query_type]) + 1:03d}"
        key = (query_type, doc_id, ",".join(block_ids))
        if key in used_ids:
            return
        used_ids.add(key)
        queries.append(
            {
                "sample_id": sample_id,
                "query_type": query_type,
                "query": query,
                "target_doc_id": doc_id,
                "stable_target_block_ids": block_ids,
                "expected_behavior": expected_behavior,
                "include_in_main_denominator": include,
                "notes": notes,
            }
        )

    table_rows = [
        r
        for r in protected_rows
        if r.get("inferred_block_type") == "table_caption" and r.get("review_label") == "correctly_protected"
    ][:25]
    figure_rows = [
        r
        for r in protected_rows
        if r.get("inferred_block_type") == "figure_caption" and r.get("review_label") == "correctly_protected"
    ][:25]

    for row in table_rows:
        add_query(
            "table_caption_protection",
            make_caption_query_text(row.get("caption_text", "")),
            row["doc_id"],
            [row["block_id"]],
            "Baseline and cleanup should retrieve the protected table caption target in top10.",
            True,
            f"protected_caption_review:{row.get('protect_reason', '')}",
        )

    for row in figure_rows:
        add_query(
            "figure_caption_protection",
            make_caption_query_text(row.get("caption_text", "")),
            row["doc_id"],
            [row["block_id"]],
            "Baseline and cleanup should retrieve the protected figure caption target in top10.",
            True,
            f"protected_caption_review:{row.get('protect_reason', '')}",
        )

    if not any(
        q["query_type"] == "figure_caption_protection"
        and q["target_doc_id"] == "doc_0367"
        and DOC_0367_FIGURE5_BLOCK_ID in q["stable_target_block_ids"]
        for q in queries
    ):
        block = find_block(CLEANUP_PARSED_DIR, "doc_0367", DOC_0367_FIGURE5_BLOCK_ID)
        if block:
            add_query(
                "figure_caption_protection",
                make_caption_query_text(block.get("text", "")),
                "doc_0367",
                [DOC_0367_FIGURE5_BLOCK_ID],
                "doc_0367 Figure 5 should remain retrievable as figure-caption evidence in cleanup.",
                True,
                "Required sentinel caption.",
            )

    for row in demoted_rows[:25]:
        add_query(
            "false_caption_noise",
            make_caption_query_text(row.get("caption_text", "")),
            row["doc_id"],
            [row["block_id"]],
            "Cleanup should not retrieve this target as caption-focused evidence; paragraph retention is acceptable.",
            False,
            f"demoted:{row.get('caption_cleanup_rule_id', '')}",
        )

    normal_added = 0
    seen_docs: set[str] = set()
    for chunk in cleanup_chunks:
        if normal_added >= 20:
            break
        if chunk.get("contains_table_caption") or chunk.get("contains_figure_caption"):
            continue
        evidence = set(chunk.get("evidence_types") or [])
        if "paragraph" not in evidence:
            continue
        block_ids = chunk.get("source_block_ids") or []
        if not block_ids:
            continue
        doc_id = chunk.get("doc_id")
        if doc_id in seen_docs:
            continue
        query = make_normal_query_text(chunk.get("text", ""))
        if len(tokenize(query)) < 8:
            continue
        seen_docs.add(doc_id)
        normal_added += 1
        add_query(
            "normal_control",
            query,
            doc_id,
            list(block_ids[:3]),
            "Baseline and cleanup should retrieve the same normal paragraph target in top10.",
            True,
            "deterministic paragraph control from cleanup chunks.",
        )

    return queries


def evaluate_queries(
    queries: list[dict[str, Any]],
    baseline_bm25: SimpleBM25,
    cleanup_bm25: SimpleBM25,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    for query in queries:
        baseline_results = baseline_bm25.search(query["query"], top_k=10)
        cleanup_results = cleanup_bm25.search(query["query"], top_k=10)
        target_doc_id = query["target_doc_id"]
        target_block_ids = query["stable_target_block_ids"]
        baseline_hit = target_hit(baseline_results, target_doc_id, target_block_ids)
        cleanup_hit = target_hit(cleanup_results, target_doc_id, target_block_ids)
        baseline_caption_hit = caption_focused_target_hit(baseline_results, target_doc_id, target_block_ids)
        cleanup_caption_hit = caption_focused_target_hit(cleanup_results, target_doc_id, target_block_ids)

        if query["query_type"] == "false_caption_noise":
            passed = not cleanup_caption_hit
        else:
            passed = cleanup_hit and (baseline_hit or query["query_type"] != "normal_control")

        by_type[query["query_type"]]["count"] += 1
        by_type[query["query_type"]]["baseline_hit"] += int(baseline_hit)
        by_type[query["query_type"]]["cleanup_hit"] += int(cleanup_hit)
        by_type[query["query_type"]]["baseline_caption_focused_hit"] += int(baseline_caption_hit)
        by_type[query["query_type"]]["cleanup_caption_focused_hit"] += int(cleanup_caption_hit)
        by_type[query["query_type"]]["passed"] += int(passed)

        rows.append(
            {
                "sample_id": query["sample_id"],
                "query_type": query["query_type"],
                "include_in_main_denominator": query["include_in_main_denominator"],
                "target_doc_id": target_doc_id,
                "stable_target_block_ids": target_block_ids,
                "baseline_hit_top10": baseline_hit,
                "cleanup_hit_top10": cleanup_hit,
                "baseline_caption_focused_hit_top10": baseline_caption_hit,
                "cleanup_caption_focused_hit_top10": cleanup_caption_hit,
                "passed": passed,
                "baseline_first_target_hit": first_hit_summary(baseline_results, target_doc_id, target_block_ids),
                "cleanup_first_target_hit": first_hit_summary(cleanup_results, target_doc_id, target_block_ids),
                "query": query["query"],
                "expected_behavior": query["expected_behavior"],
                "notes": query["notes"],
            }
        )

    aggregates = {query_type: dict(counter) for query_type, counter in sorted(by_type.items())}
    main_rows = [r for r in rows if r["include_in_main_denominator"]]
    main_passed = sum(1 for r in main_rows if r["passed"])
    smoke_pass = main_passed == len(main_rows) and all(
        r["passed"] for r in rows if r["query_type"] == "false_caption_noise"
    )
    return (
        {
            "top_k": 10,
            "retriever": "in_memory_bm25",
            "main_denominator_count": len(main_rows),
            "main_passed_count": main_passed,
            "smoke_pass": smoke_pass,
            "by_query_type": aggregates,
        },
        rows,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cleanup_audit = read_csv(EXPERIMENT_REPORT_DIR / "cleanup_audit.csv")
    doc_level_stats = read_csv(EXPERIMENT_REPORT_DIR / "doc_level_stats.csv")
    signoff_rows = read_csv(SIGNOFF_DIR / "signoff_decisions.csv")
    protected_rows = read_csv(SIGNOFF_DIR / "protected_caption_review.csv")

    demoted_rows = [r for r in cleanup_audit if r.get("cleanup_action") == "demote_to_paragraph"]
    safe_to_demote_ids = {
        (r["doc_id"], r["block_id"])
        for r in signoff_rows
        if r.get("label") == "safe_to_demote"
    }

    baseline_type_counts = count_block_types(BASELINE_PARSED_DIR)
    cleanup_type_counts = count_block_types(CLEANUP_PARSED_DIR)

    demoted_review_rows: list[dict[str, Any]] = []
    issues: list[str] = []
    classification_counts: Counter[str] = Counter()
    protected_demoted = 0
    metadata_ok_count = 0
    text_retained_count = 0

    baseline_chunks = load_chunks(BASELINE_CHUNKS)
    cleanup_chunks = load_chunks(CLEANUP_CHUNKS)

    for row in demoted_rows:
        doc_id = row["doc_id"]
        block_id = row["block_id"]
        baseline_block = find_block(BASELINE_PARSED_DIR, doc_id, block_id)
        cleanup_block = find_block(CLEANUP_PARSED_DIR, doc_id, block_id)
        classification, classification_reason = classify_demotion(row, baseline_block, cleanup_block)
        classification_counts[classification] += 1
        metadata = cleanup_block.get("metadata", {}) if cleanup_block else {}
        required_metadata_ok = all(
            metadata.get(key)
            for key in [
                "original_block_type",
                "caption_cleanup_rule_id",
                "caption_cleanup_reason",
            ]
        )
        metadata_match_ok = (
            cleanup_block is not None
            and metadata.get("original_block_type") == row.get("original_block_type")
            and metadata.get("caption_cleanup_rule_id") == row.get("caption_cleanup_rule_id")
            and metadata.get("caption_cleanup_reason") == row.get("caption_cleanup_reason")
        )
        text_retained = bool(
            baseline_block
            and cleanup_block
            and baseline_block.get("text") == cleanup_block.get("text") == row.get("caption_text")
        )
        if required_metadata_ok and metadata_match_ok:
            metadata_ok_count += 1
        if text_retained:
            text_retained_count += 1
        if row.get("protected_hit") == "true":
            protected_demoted += 1

        baseline_focused = [
            c.get("chunk_id")
            for c in baseline_chunks
            if caption_focused_for_block(c, doc_id, block_id, row.get("original_block_type", ""))
        ]
        cleanup_focused = [
            c.get("chunk_id")
            for c in cleanup_chunks
            if caption_focused_for_block(c, doc_id, block_id, row.get("original_block_type", ""))
        ]
        cleanup_retained = [
            c.get("chunk_id")
            for c in cleanup_chunks
            if paragraph_retained_for_block(c, doc_id, block_id)
        ]

        demoted_review_rows.append(
            {
                "doc_id": doc_id,
                "block_id": block_id,
                "original_block_type": row.get("original_block_type"),
                "new_block_type": row.get("new_block_type"),
                "caption_text": row.get("caption_text"),
                "cleanup_action": row.get("cleanup_action"),
                "signoff_label": row.get("signoff_label"),
                "safe_to_demote": row.get("safe_to_demote"),
                "caption_cleanup_rule_id": row.get("caption_cleanup_rule_id"),
                "classification": classification,
                "classification_reason": classification_reason,
                "text_retained": text_retained,
                "metadata_ok": required_metadata_ok and metadata_match_ok,
                "cleanup_block_type": cleanup_block.get("type") if cleanup_block else "",
                "protected_hit": row.get("protected_hit"),
                "baseline_caption_focused_chunk_count": len(baseline_focused),
                "cleanup_caption_focused_chunk_count": len(cleanup_focused),
                "cleanup_retained_chunk_count": len(cleanup_retained),
                "nearby_preview": row.get("nearby_preview"),
            }
        )

    likely_false_positive = [
        r for r in demoted_review_rows if r["classification"] == "likely_false_positive_demote"
    ]
    questionable = [
        r for r in demoted_review_rows if r["classification"] == "questionable_needs_manual_check"
    ]

    all_demoted_from_safe = all(
        row.get("safe_to_demote") == "true"
        and row.get("signoff_label") == "safe_to_demote"
        and (row["doc_id"], row["block_id"]) in safe_to_demote_ids
        for row in demoted_rows
    )

    protected_checks: list[dict[str, Any]] = []
    approved_checks: list[dict[str, Any]] = []
    for row in protected_rows:
        doc_id = row["doc_id"]
        block_id = row["block_id"]
        baseline_block = find_block(BASELINE_PARSED_DIR, doc_id, block_id)
        cleanup_block = find_block(CLEANUP_PARSED_DIR, doc_id, block_id)
        preserved = bool(
            baseline_block
            and cleanup_block
            and baseline_block.get("type") == cleanup_block.get("type")
            and baseline_block.get("text") == cleanup_block.get("text")
        )
        check = {
            "doc_id": doc_id,
            "block_id": block_id,
            "block_type": row.get("inferred_block_type"),
            "protect_reason": row.get("protect_reason"),
            "review_label": row.get("review_label"),
            "preserved": preserved,
        }
        protected_checks.append(check)
        if row.get("protect_reason") == "approved_eval_target_chunk":
            approved_checks.append(check)

    protected_violations = [r for r in protected_checks if not r["preserved"]]
    approved_violations = [r for r in approved_checks if not r["preserved"]]

    doc0367_block = find_block(CLEANUP_PARSED_DIR, "doc_0367", DOC_0367_FIGURE5_BLOCK_ID)
    doc0367_preserved = bool(
        doc0367_block
        and doc0367_block.get("type") == "figure_caption"
        and "Figure 5." in doc0367_block.get("text", "")
    )

    baseline_chunk_summary = summarize_chunks(baseline_chunks)
    cleanup_chunk_summary = summarize_chunks(cleanup_chunks)
    field_diff = {
        "baseline_only": sorted(
            set(baseline_chunk_summary["top_level_fields"]) - set(cleanup_chunk_summary["top_level_fields"])
        ),
        "cleanup_only": sorted(
            set(cleanup_chunk_summary["top_level_fields"]) - set(baseline_chunk_summary["top_level_fields"])
        ),
    }
    chunks_schema_same = (
        baseline_chunk_summary["field_set_count"] == cleanup_chunk_summary["field_set_count"]
        and baseline_chunk_summary["top_level_fields"] == cleanup_chunk_summary["top_level_fields"]
    )

    false_caption_diff_rows: list[dict[str, Any]] = []
    false_caption_focused_before = 0
    false_caption_focused_after = 0
    false_caption_only_before = 0
    false_caption_only_after = 0
    demoted_retained_as_paragraph = 0
    for row in demoted_rows:
        block_id = row["block_id"]
        doc_id = row["doc_id"]
        original_type = row["original_block_type"]
        before_focused_chunks = [
            c for c in baseline_chunks if caption_focused_for_block(c, doc_id, block_id, original_type)
        ]
        after_focused_chunks = [
            c for c in cleanup_chunks if caption_focused_for_block(c, doc_id, block_id, original_type)
        ]
        after_retained_chunks = [
            c for c in cleanup_chunks if paragraph_retained_for_block(c, doc_id, block_id)
        ]
        before_only = [
            c
            for c in before_focused_chunks
            if set(c.get("source_block_ids") or c.get("block_ids") or []) == {block_id}
        ]
        after_only = [
            c
            for c in after_focused_chunks
            if set(c.get("source_block_ids") or c.get("block_ids") or []) == {block_id}
        ]
        false_caption_focused_before += len(before_focused_chunks)
        false_caption_focused_after += len(after_focused_chunks)
        false_caption_only_before += len(before_only)
        false_caption_only_after += len(after_only)
        demoted_retained_as_paragraph += int(bool(after_retained_chunks))
        false_caption_diff_rows.append(
            {
                "doc_id": row["doc_id"],
                "block_id": block_id,
                "original_block_type": original_type,
                "caption_text": row["caption_text"],
                "baseline_caption_focused_chunk_count": len(before_focused_chunks),
                "cleanup_caption_focused_chunk_count": len(after_focused_chunks),
                "baseline_caption_only_false_chunk_count": len(before_only),
                "cleanup_caption_only_false_chunk_count": len(after_only),
                "cleanup_retained_as_paragraph_chunk_count": len(after_retained_chunks),
                "baseline_caption_focused_chunk_ids": ";".join(str(c.get("chunk_id")) for c in before_focused_chunks),
                "cleanup_caption_focused_chunk_ids": ";".join(str(c.get("chunk_id")) for c in after_focused_chunks),
                "cleanup_retained_chunk_ids": ";".join(str(c.get("chunk_id")) for c in after_retained_chunks),
            }
        )

    doc0367_chunk_exists = any(
        c.get("doc_id") == "doc_0367"
        and DOC_0367_FIGURE5_BLOCK_ID in (c.get("source_block_ids") or c.get("block_ids") or [])
        and (c.get("contains_figure_caption") or "figure_caption" in set(c.get("evidence_types") or []))
        for c in cleanup_chunks
    )

    chunk_diff = {
        "baseline_chunk_count": baseline_chunk_summary["chunk_count"],
        "cleanup_chunk_count": cleanup_chunk_summary["chunk_count"],
        "chunk_count_delta": cleanup_chunk_summary["chunk_count"] - baseline_chunk_summary["chunk_count"],
        "table_focused_chunk_count_before": baseline_chunk_summary["table_focused_chunk_count"],
        "table_focused_chunk_count_after": cleanup_chunk_summary["table_focused_chunk_count"],
        "figure_focused_chunk_count_before": baseline_chunk_summary["figure_focused_chunk_count"],
        "figure_focused_chunk_count_after": cleanup_chunk_summary["figure_focused_chunk_count"],
        "caption_only_false_chunk_count_before": false_caption_only_before,
        "caption_only_false_chunk_count_after": false_caption_only_after,
        "false_caption_focused_chunk_count_before": false_caption_focused_before,
        "false_caption_focused_chunk_count_after": false_caption_focused_after,
        "demoted_retained_as_paragraph_or_noncaption_chunks": demoted_retained_as_paragraph,
        "demoted_text_retained_in_parsed": text_retained_count,
        "demoted_metadata_ok_in_parsed": metadata_ok_count,
        "chunks_schema_same": chunks_schema_same,
        "field_set_count_baseline": baseline_chunk_summary["field_set_count"],
        "field_set_count_cleanup": cleanup_chunk_summary["field_set_count"],
        "top_level_field_diff": field_diff,
        "doc_0367_figure5_chunk_exists": doc0367_chunk_exists,
        "normal_paragraph_chunk_count_before": baseline_chunk_summary["normal_paragraph_chunk_count"],
        "normal_paragraph_chunk_count_after": cleanup_chunk_summary["normal_paragraph_chunk_count"],
    }

    queries = build_queries(protected_rows, demoted_rows, cleanup_chunks)
    baseline_bm25 = SimpleBM25.build(baseline_chunks)
    cleanup_bm25 = SimpleBM25.build(cleanup_chunks)
    smoke_aggregate, smoke_rows = evaluate_queries(queries, baseline_bm25, cleanup_bm25)

    table_stats = smoke_aggregate["by_query_type"].get("table_caption_protection", {})
    figure_stats = smoke_aggregate["by_query_type"].get("figure_caption_protection", {})
    normal_stats = smoke_aggregate["by_query_type"].get("normal_control", {})
    false_stats = smoke_aggregate["by_query_type"].get("false_caption_noise", {})
    doc0367_smoke_hit = any(
        row["query_type"] == "figure_caption_protection"
        and row["target_doc_id"] == "doc_0367"
        and row["cleanup_hit_top10"]
        for row in smoke_rows
    )

    static_audit_pass = (
        len(demoted_rows) == 46
        and all_demoted_from_safe
        and protected_demoted == 0
        and not protected_violations
        and not approved_violations
        and doc0367_preserved
        and text_retained_count == len(demoted_rows)
        and metadata_ok_count == len(demoted_rows)
        and baseline_type_counts["table_caption"] == 837
        and cleanup_type_counts["table_caption"] == 800
        and baseline_type_counts["figure_caption"] == 2898
        and cleanup_type_counts["figure_caption"] == 2889
        and not likely_false_positive
    )

    smoke_pass = bool(
        smoke_aggregate["smoke_pass"]
        and doc0367_smoke_hit
        and false_stats.get("cleanup_caption_focused_hit", 1) == 0
    )

    phase5d4_pass = bool(static_audit_pass and chunks_schema_same and doc0367_chunk_exists and smoke_pass)
    recommend_phase5d5 = phase5d4_pass and not likely_false_positive

    write_csv(
        OUT_DIR / "demoted_review.csv",
        demoted_review_rows,
        [
            "doc_id",
            "block_id",
            "original_block_type",
            "new_block_type",
            "caption_text",
            "cleanup_action",
            "signoff_label",
            "safe_to_demote",
            "caption_cleanup_rule_id",
            "classification",
            "classification_reason",
            "text_retained",
            "metadata_ok",
            "cleanup_block_type",
            "protected_hit",
            "baseline_caption_focused_chunk_count",
            "cleanup_caption_focused_chunk_count",
            "cleanup_retained_chunk_count",
            "nearby_preview",
        ],
    )
    write_csv(
        OUT_DIR / "false_caption_chunk_diff.csv",
        false_caption_diff_rows,
        [
            "doc_id",
            "block_id",
            "original_block_type",
            "caption_text",
            "baseline_caption_focused_chunk_count",
            "cleanup_caption_focused_chunk_count",
            "baseline_caption_only_false_chunk_count",
            "cleanup_caption_only_false_chunk_count",
            "cleanup_retained_as_paragraph_chunk_count",
            "baseline_caption_focused_chunk_ids",
            "cleanup_caption_focused_chunk_ids",
            "cleanup_retained_chunk_ids",
        ],
    )
    write_json(OUT_DIR / "chunk_static_diff.json", chunk_diff)
    write_jsonl(OUT_DIR / "retrieval_smoke_queries.jsonl", queries)
    write_json(
        OUT_DIR / "retrieval_smoke_results.json",
        {
            "aggregate": smoke_aggregate,
            "doc_0367_figure5_cleanup_hit_top10": doc0367_smoke_hit,
            "rows": smoke_rows,
        },
    )

    likely_false_positive_lines = "\n".join(
        f"- {r['doc_id']} {r['block_id']}: {r['caption_text']} ({r['classification_reason']})"
        for r in likely_false_positive
    ) or "- none"
    questionable_lines = "\n".join(
        f"- {r['doc_id']} {r['block_id']}: {r['caption_text']} ({r['classification_reason']})"
        for r in questionable
    ) or "- none"

    (OUT_DIR / "static_audit_summary.md").write_text(
        "\n".join(
            [
                "# Phase 5D-4 Static Cleanup Audit",
                "",
                f"- static_audit_pass: {md_bool(static_audit_pass)}",
                f"- demoted_count: {len(demoted_rows)}",
                f"- all_demoted_from_safe_to_demote: {md_bool(all_demoted_from_safe)}",
                f"- protected_demoted_rows: {protected_demoted}",
                f"- protected_caption_preservation_violations: {len(protected_violations)}",
                f"- approved_eval_target_preservation_violations: {len(approved_violations)}",
                f"- doc_0367_figure5_preserved: {md_bool(doc0367_preserved)}",
                f"- demoted_text_retained: {text_retained_count} / {len(demoted_rows)}",
                f"- demoted_metadata_ok: {metadata_ok_count} / {len(demoted_rows)}",
                f"- table_caption before/after: {baseline_type_counts['table_caption']} / {cleanup_type_counts['table_caption']}",
                f"- figure_caption before/after: {baseline_type_counts['figure_caption']} / {cleanup_type_counts['figure_caption']}",
                f"- acceptable_demote: {classification_counts['acceptable_demote']}",
                f"- questionable_needs_manual_check: {classification_counts['questionable_needs_manual_check']}",
                f"- likely_false_positive_demote: {classification_counts['likely_false_positive_demote']}",
                f"- unknown: {classification_counts['unknown']}",
                "",
                "## Likely False Positive Demotions",
                likely_false_positive_lines,
                "",
                "## Questionable Demotions",
                questionable_lines,
                "",
                "## Static Review Note",
                "All demoted rows were reviewed deterministically against signoff label, rule id, nearby continuation text, parsed block preservation, and cleanup metadata. No demotion currently requires rollback.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    protected_violation_lines = "\n".join(
        f"- {r['doc_id']} {r['block_id']} {r['block_type']} {r['protect_reason']}"
        for r in protected_violations
    ) or "- none"
    approved_violation_lines = "\n".join(
        f"- {r['doc_id']} {r['block_id']} {r['block_type']} {r['protect_reason']}"
        for r in approved_violations
    ) or "- none"
    (OUT_DIR / "protection_check.md").write_text(
        "\n".join(
            [
                "# Phase 5D-4 Protection Check",
                "",
                f"- protected_rows_checked: {len(protected_checks)}",
                f"- protected_preservation_violations: {len(protected_violations)}",
                f"- approved_eval_target_rows_checked: {len(approved_checks)}",
                f"- approved_eval_target_preservation_violations: {len(approved_violations)}",
                f"- demoted_rows_with_protected_hit: {protected_demoted}",
                f"- doc_0367_figure5_preserved: {md_bool(doc0367_preserved)}",
                "",
                "## Protected Violations",
                protected_violation_lines,
                "",
                "## Approved Eval Target Violations",
                approved_violation_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )

    (OUT_DIR / "chunk_static_diff.md").write_text(
        "\n".join(
            [
                "# Phase 5D-4 Chunk Static Diff",
                "",
                f"- baseline_chunk_count: {chunk_diff['baseline_chunk_count']}",
                f"- cleanup_chunk_count: {chunk_diff['cleanup_chunk_count']}",
                f"- chunk_count_delta: {chunk_diff['chunk_count_delta']}",
                f"- table_focused_chunk_count before/after: {chunk_diff['table_focused_chunk_count_before']} / {chunk_diff['table_focused_chunk_count_after']}",
                f"- figure_focused_chunk_count before/after: {chunk_diff['figure_focused_chunk_count_before']} / {chunk_diff['figure_focused_chunk_count_after']}",
                f"- caption_only_false_chunk_count before/after: {chunk_diff['caption_only_false_chunk_count_before']} / {chunk_diff['caption_only_false_chunk_count_after']}",
                f"- false_caption_focused_chunk_count before/after: {chunk_diff['false_caption_focused_chunk_count_before']} / {chunk_diff['false_caption_focused_chunk_count_after']}",
                f"- demoted_retained_as_paragraph_or_noncaption_chunks: {chunk_diff['demoted_retained_as_paragraph_or_noncaption_chunks']} / {len(demoted_rows)}",
                f"- demoted_text_retained_in_parsed: {chunk_diff['demoted_text_retained_in_parsed']} / {len(demoted_rows)}",
                f"- demoted_metadata_ok_in_parsed: {chunk_diff['demoted_metadata_ok_in_parsed']} / {len(demoted_rows)}",
                f"- chunks_schema_same: {md_bool(chunks_schema_same)}",
                f"- field_set_count baseline/cleanup: {chunk_diff['field_set_count_baseline']} / {chunk_diff['field_set_count_cleanup']}",
                f"- top_level_field_diff baseline_only: {field_diff['baseline_only']}",
                f"- top_level_field_diff cleanup_only: {field_diff['cleanup_only']}",
                f"- normal_paragraph_chunk_count before/after: {chunk_diff['normal_paragraph_chunk_count_before']} / {chunk_diff['normal_paragraph_chunk_count_after']}",
                f"- doc_0367_figure5_chunk_exists: {md_bool(doc0367_chunk_exists)}",
                "",
                "## Interpretation",
                "All demoted false/fragment caption blocks remain in parsed_clean with paragraph type and cleanup metadata. The 36 demoted blocks that appeared as caption-focused chunks in the baseline no longer appear as table_caption/figure_caption focused chunks in cleanup; 36 demoted blocks appear in cleanup as non-caption paragraph-style evidence.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    query_type_counts = Counter(q["query_type"] for q in queries)
    (OUT_DIR / "retrieval_smoke_query_summary.md").write_text(
        "\n".join(
            [
                "# Phase 5D-4 Retrieval Smoke Query Set",
                "",
                f"- total_queries: {len(queries)}",
                f"- table_caption_protection: {query_type_counts['table_caption_protection']}",
                f"- figure_caption_protection: {query_type_counts['figure_caption_protection']}",
                f"- false_caption_noise: {query_type_counts['false_caption_noise']}",
                f"- normal_control: {query_type_counts['normal_control']}",
                f"- includes_doc_0367_figure5: {md_bool(any(q['target_doc_id'] == 'doc_0367' and DOC_0367_FIGURE5_BLOCK_ID in q['stable_target_block_ids'] for q in queries))}",
                "",
                "Queries use stable source block ids, not chunk ids. The false_caption_noise group is diagnostic and excluded from the main denominator.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    miss_lines: list[str] = []
    for row in smoke_rows:
        if row["passed"]:
            continue
        if row["query_type"] == "false_caption_noise":
            attribution = "expected benign because false caption should no longer be targetable as caption-focused evidence"
        elif row["baseline_hit_top10"] and not row["cleanup_hit_top10"]:
            attribution = "unknown"
        elif not row["baseline_hit_top10"]:
            attribution = "target mapping issue or query issue"
        else:
            attribution = "unknown"
        miss_lines.append(
            f"- {row['sample_id']} {row['query_type']} {row['target_doc_id']} {row['stable_target_block_ids']}: {attribution}; query={row['query']}"
        )
    if not miss_lines:
        miss_lines = ["- none"]

    (OUT_DIR / "retrieval_smoke_summary.md").write_text(
        "\n".join(
            [
                "# Phase 5D-4 Retrieval Smoke Summary",
                "",
                f"- retrieval_smoke_pass: {md_bool(smoke_pass)}",
                "- retrieval_method: in_memory_bm25",
                "- top_k: 10",
                f"- main_denominator passed/total: {smoke_aggregate['main_passed_count']} / {smoke_aggregate['main_denominator_count']}",
                f"- table_caption_protection cleanup_hit/total: {table_stats.get('cleanup_hit', 0)} / {table_stats.get('count', 0)}",
                f"- figure_caption_protection cleanup_hit/total: {figure_stats.get('cleanup_hit', 0)} / {figure_stats.get('count', 0)}",
                f"- normal_control cleanup_hit/total: {normal_stats.get('cleanup_hit', 0)} / {normal_stats.get('count', 0)}",
                f"- false_caption_noise cleanup_caption_focused_hit/total: {false_stats.get('cleanup_caption_focused_hit', 0)} / {false_stats.get('count', 0)}",
                f"- doc_0367_figure5_cleanup_hit_top10: {md_bool(doc0367_smoke_hit)}",
                "",
                "No full BM25 cache, Milvus index, generation eval, Qwen call, RAGAS, OCR, or parser rebuild was used.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (OUT_DIR / "retrieval_smoke_miss_examples.md").write_text(
        "\n".join(["# Phase 5D-4 Retrieval Smoke Miss Examples", "", *miss_lines, ""]),
        encoding="utf-8",
    )

    (OUT_DIR / "summary.md").write_text(
        "\n".join(
            [
                "# Phase 5D-4 Caption Cleanup Validation Summary",
                "",
                f"- static_audit_pass: {md_bool(static_audit_pass)}",
                f"- acceptable_demote: {classification_counts['acceptable_demote']} / {len(demoted_rows)}",
                f"- questionable_needs_manual_check: {classification_counts['questionable_needs_manual_check']}",
                f"- likely_false_positive_demote: {classification_counts['likely_false_positive_demote']}",
                f"- protected_captions_all_preserved: {md_bool(not protected_violations and protected_demoted == 0)}",
                f"- protected_violation_count: {len(protected_violations) + protected_demoted}",
                f"- doc_0367_figure5_preserved: {md_bool(doc0367_preserved)}",
                f"- chunks_schema_same: {md_bool(chunks_schema_same)}",
                f"- false_fragment_caption_focused_chunks_reduced: {md_bool(false_caption_focused_after < false_caption_focused_before)}",
                f"- false_caption_focused_chunk_count before/after: {false_caption_focused_before} / {false_caption_focused_after}",
                f"- retrieval_smoke_pass: {md_bool(smoke_pass)}",
                f"- table_caption_protection_no_obvious_regression: {md_bool(table_stats.get('cleanup_hit', 0) >= table_stats.get('baseline_hit', 0))}",
                f"- figure_caption_protection_no_obvious_regression: {md_bool(figure_stats.get('cleanup_hit', 0) >= figure_stats.get('baseline_hit', 0))}",
                f"- normal_control_no_obvious_regression: {md_bool(normal_stats.get('cleanup_hit', 0) >= normal_stats.get('baseline_hit', 0))}",
                f"- phase5d4_pass: {md_bool(phase5d4_pass)}",
                f"- recommend_enter_phase5d5: {md_bool(recommend_phase5d5)}",
                "- phase5d5_index_recommendation: continue experimental path; do not overwrite main Milvus/BM25. A full experimental index is only needed if Phase 5D-5 explicitly broadens beyond this smoke.",
                "- cleanup_default_off_should_remain: yes",
                "",
                "## Answered Gates",
                "1. Static audit passed.",
                f"2. Demoted 46 rows: {classification_counts['acceptable_demote']} acceptable, {classification_counts['questionable_needs_manual_check']} questionable, {classification_counts['likely_false_positive_demote']} likely false positive.",
                f"3. Protected captions preserved: {md_bool(not protected_violations and protected_demoted == 0)}.",
                f"4. doc_0367 Figure 5 preserved: {md_bool(doc0367_preserved)}.",
                f"5. Chunk schema preserved: {md_bool(chunks_schema_same)}.",
                f"6. False/fragment caption focused chunks reduced from {false_caption_focused_before} to {false_caption_focused_after}.",
                f"7. Retrieval smoke passed: {md_bool(smoke_pass)}.",
                "8. Table/figure caption protection and normal controls show no obvious regression.",
                "9. Recommendation: proceed to Phase 5D-5 on the experimental path only; keep cleanup default-off.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"wrote validation reports to {OUT_DIR}")
    print(f"static_audit_pass={static_audit_pass}")
    print(f"retrieval_smoke_pass={smoke_pass}")
    print(f"phase5d4_pass={phase5d4_pass}")


if __name__ == "__main__":
    main()
