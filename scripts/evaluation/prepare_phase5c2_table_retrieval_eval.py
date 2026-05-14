#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._'′+/\-()α-ωΑ-ΩΔδ]{2,}")
STOP = {
    "table", "figure", "summary", "summarized", "reports", "lists", "where", "which",
    "from", "with", "that", "this", "were", "have", "been", "protein", "proteins",
    "gene", "genes", "strain", "strains", "source", "function", "conditions",
    "medium", "activity", "values", "different", "using", "used", "study",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic Phase 5C-2 retrieval eval queries.")
    parser.add_argument("--selected-docs", default="reports/phase5_table_audit/selected_docs.csv")
    parser.add_argument("--association-audit", default="reports/phase5c1_table_preservation/association_audit.csv")
    parser.add_argument("--phase4-approved-eval", default="reports/table_figure_retrieval_eval/phase4e3_eval_set_approved/eval_set.jsonl")
    parser.add_argument("--enhanced-chunks", default="/tmp/biorag_phase5c1_enhanced/chunks/chunks.jsonl")
    parser.add_argument("--baseline-chunks", default="/tmp/biorag_phase5c2_baseline/chunks.jsonl")
    parser.add_argument("--output-dir", default="reports/phase5c2_table_retrieval_ab")
    parser.add_argument("--table-content-count", type=int, default=40)
    parser.add_argument("--caption-count", type=int, default=30)
    parser.add_argument("--normal-count", type=int, default=30)
    parser.add_argument("--diagnostic-count", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_doc_ids = load_selected_doc_ids(Path(args.selected_docs))
    selected_set = set(selected_doc_ids)
    baseline_chunks = list(iter_jsonl(Path(args.baseline_chunks)))
    enhanced_chunks = list(iter_jsonl(Path(args.enhanced_chunks)))
    baseline_by_doc = group_by_doc(baseline_chunks)
    enhanced_by_doc = group_by_doc(enhanced_chunks)
    baseline_by_id = {c["chunk_id"]: c for c in baseline_chunks}
    enhanced_by_id = {c["chunk_id"]: c for c in enhanced_chunks}
    assoc_rows = load_accepted_associations(Path(args.association_audit), selected_set)

    queries: list[dict[str, Any]] = []
    queries.extend(build_table_content_queries(
        assoc_rows=assoc_rows,
        baseline_by_doc=baseline_by_doc,
        enhanced_by_doc=enhanced_by_doc,
        limit=args.table_content_count,
    ))
    queries.extend(build_caption_queries(
        approved_path=Path(args.phase4_approved_eval),
        selected_doc_ids=selected_set,
        baseline_by_id=baseline_by_id,
        enhanced_by_id=enhanced_by_id,
        enhanced_by_doc=enhanced_by_doc,
        limit=args.caption_count,
    ))
    queries.extend(build_normal_queries(
        baseline_chunks=baseline_chunks,
        enhanced_by_id=enhanced_by_id,
        enhanced_by_doc=enhanced_by_doc,
        limit=args.normal_count,
    ))
    queries.extend(build_diagnostic_queries(assoc_rows, enhanced_by_doc, limit=args.diagnostic_count))

    out_jsonl = output_dir / "eval_queries.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for item in queries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = summarize_queries(queries)
    (output_dir / "eval_query_summary.md").write_text(summary, encoding="utf-8")
    print(f"Wrote {out_jsonl}")
    print(f"Wrote {output_dir / 'eval_query_summary.md'}")


def load_selected_doc_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [row["doc_id"] for row in csv.DictReader(f) if row.get("doc_id")]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def group_by_doc(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        grouped[str(chunk.get("doc_id", ""))].append(chunk)
    return grouped


def load_accepted_associations(path: Path, selected_doc_ids: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("accepted_or_rejected") == "accepted" and row.get("doc_id") in selected_doc_ids:
                rows.append(row)
    return rows


def block_ids(chunk: dict[str, Any]) -> set[str]:
    return set(chunk.get("source_block_ids") or []) | set(chunk.get("block_ids") or [])


def find_chunk_by_block(chunks: list[dict[str, Any]], block_id: str) -> dict[str, Any] | None:
    for chunk in chunks:
        if block_id in block_ids(chunk):
            return chunk
    return None


def find_caption_chunk(chunks: list[dict[str, Any]], caption_block_id: str, anchors: list[str]) -> dict[str, Any] | None:
    exact = find_chunk_by_block(chunks, caption_block_id)
    if exact:
        return exact
    scored: list[tuple[int, dict[str, Any]]] = []
    anchor_l = [a.lower() for a in anchors]
    for chunk in chunks:
        if not chunk.get("contains_table_caption"):
            continue
        text = (chunk.get("text", "") + "\n" + chunk.get("retrieval_text", "")).lower()
        score = sum(1 for anchor in anchor_l if anchor and anchor in text)
        if score:
            scored.append((score, chunk))
    scored.sort(key=lambda item: (-item[0], item[1].get("chunk_id", "")))
    return scored[0][1] if scored else None


def build_table_content_queries(
    assoc_rows: list[dict[str, str]],
    baseline_by_doc: dict[str, list[dict[str, Any]]],
    enhanced_by_doc: dict[str, list[dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_chunks: Counter[str] = Counter()
    for row in assoc_rows:
        doc_id = row["doc_id"]
        anchors = anchor_terms(row.get("associated_text_preview", ""), row.get("caption_text_preview", ""))
        if len(anchors) < 2:
            continue
        enhanced = find_chunk_by_block(enhanced_by_doc[doc_id], row.get("associated_block_id", ""))
        if not enhanced:
            continue
        baseline = find_caption_chunk(baseline_by_doc[doc_id], row.get("table_caption_block_id", ""), anchors)
        if seen_chunks[enhanced["chunk_id"]] >= 2:
            continue
        seen_chunks[enhanced["chunk_id"]] += 1
        candidates.append({
            "sample_id": f"p5c2_table_content_{len(candidates) + 1:04d}",
            "query_type": "table_content",
            "query": table_content_query(anchors, row.get("associated_text_preview", ""), row.get("caption_text_preview", "")),
            "target_doc_id": doc_id,
            "target_chunk_id_baseline": baseline.get("chunk_id", "") if baseline else "",
            "target_chunk_id_enhanced": enhanced.get("chunk_id", ""),
            "target_associated_block_id": row.get("associated_block_id", ""),
            "target_caption_block_id": row.get("table_caption_block_id", ""),
            "anchor_terms": anchors,
            "expected_gain_source": "table_related_association",
            "include_in_main_denominator": True,
            "notes": f"association_confidence={row.get('association_confidence', '')}; rule_hits={row.get('rule_hits', '')}",
        })
        if len(candidates) >= limit:
            break
    return candidates


def table_content_query(anchors: list[str], associated_text: str, caption_text: str) -> str:
    joined = " ".join(anchors[:3])
    lower = f"{associated_text} {caption_text}".lower()
    if any(term in lower for term in ("strain", "plasmid", "primer", "genotype", "source", "orf")):
        return f"Which table lists {joined} information?"
    if any(term in lower for term in ("yield", "titer", "activity", "condition", "medium", "biomass", "rate")):
        return f"Which table reports {joined} measurements?"
    if any(term in lower for term in ("gene", "protein", "enzyme", "mutant", "homolog")):
        return f"Where are {joined} summarized in a table?"
    return f"Which table contains {joined}?"


def build_caption_queries(
    approved_path: Path,
    selected_doc_ids: set[str],
    baseline_by_id: dict[str, dict[str, Any]],
    enhanced_by_id: dict[str, dict[str, Any]],
    enhanced_by_doc: dict[str, list[dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    if not approved_path.exists():
        return []
    queries: list[dict[str, Any]] = []
    for sample in iter_jsonl(approved_path):
        if sample.get("target_doc_id") not in selected_doc_ids:
            continue
        if sample.get("sample_type") != "table" or not sample.get("approved"):
            continue
        if sample.get("include_in_main_denominator") is False:
            continue
        target_id = str(sample.get("target_chunk_id", ""))
        anchors = [str(x) for x in sample.get("anchor_terms") or [] if str(x).strip()][:4]
        enhanced_target = enhanced_by_id.get(target_id)
        if not enhanced_target:
            enhanced_target = find_caption_chunk(enhanced_by_doc[str(sample.get("target_doc_id", ""))], "", anchors)
        queries.append({
            "sample_id": f"p5c2_caption_level_{len(queries) + 1:04d}",
            "query_type": "caption_level_table",
            "query": str(sample.get("query", "")),
            "target_doc_id": str(sample.get("target_doc_id", "")),
            "target_chunk_id_baseline": target_id if target_id in baseline_by_id else "",
            "target_chunk_id_enhanced": enhanced_target.get("chunk_id", "") if enhanced_target else "",
            "target_associated_block_id": "",
            "target_caption_block_id": "",
            "anchor_terms": anchors,
            "expected_gain_source": "caption_baseline",
            "include_in_main_denominator": True,
            "notes": f"source_sample_id={sample.get('sample_id', '')}",
        })
        if len(queries) >= limit:
            break
    return queries


def build_normal_queries(
    baseline_chunks: list[dict[str, Any]],
    enhanced_by_id: dict[str, dict[str, Any]],
    enhanced_by_doc: dict[str, list[dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    candidates = [
        c for c in baseline_chunks
        if is_normal_paragraph(c) and len(c.get("text", "")) >= 250
    ]
    candidates.sort(key=lambda c: (c.get("doc_id", ""), c.get("chunk_index", 0), c.get("chunk_id", "")))
    queries: list[dict[str, Any]] = []
    per_doc: Counter[str] = Counter()
    for chunk in candidates:
        doc_id = str(chunk.get("doc_id", ""))
        if per_doc[doc_id] >= 1 and len(queries) < 25:
            continue
        anchors = anchor_terms(chunk.get("text", ""), chunk.get("title", ""))
        if len(anchors) < 2:
            continue
        enhanced = enhanced_by_id.get(chunk["chunk_id"]) or match_by_blocks(enhanced_by_doc[doc_id], chunk)
        queries.append({
            "sample_id": f"p5c2_normal_control_{len(queries) + 1:04d}",
            "query_type": "normal_control",
            "query": normal_query(anchors, chunk),
            "target_doc_id": doc_id,
            "target_chunk_id_baseline": chunk.get("chunk_id", ""),
            "target_chunk_id_enhanced": enhanced.get("chunk_id", "") if enhanced else "",
            "target_associated_block_id": "",
            "target_caption_block_id": "",
            "anchor_terms": anchors[:4],
            "expected_gain_source": "normal_control",
            "include_in_main_denominator": True,
            "notes": "paragraph control from selected docs",
        })
        per_doc[doc_id] += 1
        if len(queries) >= limit:
            break
    return queries


def is_normal_paragraph(chunk: dict[str, Any]) -> bool:
    if chunk.get("contains_table_text") or chunk.get("contains_table_caption") or chunk.get("contains_figure_caption"):
        return False
    if chunk.get("contains_references") or chunk.get("contains_metadata"):
        return False
    types = set(chunk.get("evidence_types") or chunk.get("block_types") or [])
    return "paragraph" in types


def match_by_blocks(chunks: list[dict[str, Any]], target: dict[str, Any]) -> dict[str, Any] | None:
    wanted = block_ids(target)
    if not wanted:
        return None
    scored = []
    for chunk in chunks:
        overlap = len(wanted & block_ids(chunk))
        if overlap:
            scored.append((overlap, chunk))
    scored.sort(key=lambda item: (-item[0], item[1].get("chunk_id", "")))
    return scored[0][1] if scored else None


def normal_query(anchors: list[str], chunk: dict[str, Any]) -> str:
    section = str(chunk.get("section") or "the paper")
    return f"What does {section} report about {' '.join(anchors[:3])}?"


def build_diagnostic_queries(
    assoc_rows: list[dict[str, str]],
    enhanced_by_doc: dict[str, list[dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for row in reversed(assoc_rows):
        anchors = anchor_terms(row.get("associated_text_preview", ""), row.get("caption_text_preview", ""))
        if len(anchors) < 1:
            continue
        enhanced = find_chunk_by_block(enhanced_by_doc[row["doc_id"]], row.get("associated_block_id", ""))
        if not enhanced:
            continue
        queries.append({
            "sample_id": f"p5c2_diagnostic_{len(queries) + 1:04d}",
            "query_type": "diagnostic_only",
            "query": f"Where is {anchors[0]} mentioned?",
            "target_doc_id": row["doc_id"],
            "target_chunk_id_baseline": "",
            "target_chunk_id_enhanced": enhanced.get("chunk_id", ""),
            "target_associated_block_id": row.get("associated_block_id", ""),
            "target_caption_block_id": row.get("table_caption_block_id", ""),
            "anchor_terms": anchors[:3],
            "expected_gain_source": "table_related_association",
            "include_in_main_denominator": False,
            "notes": "ambiguous diagnostic probe; excluded from main denominator",
        })
        if len(queries) >= limit:
            break
    return queries


def anchor_terms(primary: str, secondary: str = "") -> list[str]:
    text = f"{primary} {secondary}"
    terms: list[str] = []
    seen: set[str] = set()
    for raw in TOKEN_RE.findall(text):
        token = raw.strip(".,;:()[]{}")
        key = token.lower()
        if key in STOP or len(key) < 3:
            continue
        informative = any(ch.isdigit() for ch in token) or any(ch.isupper() for ch in token) or "-" in token or len(token) >= 6
        if not informative:
            continue
        if key not in seen:
            seen.add(key)
            terms.append(token)
        if len(terms) >= 5:
            break
    return terms


def summarize_queries(queries: list[dict[str, Any]]) -> str:
    by_type = Counter(q["query_type"] for q in queries)
    main_by_type = Counter(q["query_type"] for q in queries if q.get("include_in_main_denominator"))
    lines = [
        "# Phase 5C-2 Eval Query Summary",
        "",
        f"- total queries: {len(queries)}",
        f"- main denominator queries: {sum(main_by_type.values())}",
        "",
        "## By Query Type",
        "",
    ]
    for qtype in sorted(by_type):
        lines.append(f"- {qtype}: {by_type[qtype]} total, {main_by_type[qtype]} main")
    lines.extend([
        "",
        "## Notes",
        "",
        "- Queries were generated deterministically from selected 50-doc artifacts.",
        "- table_content targets enhanced chunks containing accepted table-related associated blocks.",
        "- diagnostic_only probes are excluded from the main denominator.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
