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
    "cells", "cell", "results", "analysis", "shown", "showed", "effect",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic Phase 5C-3 retrieval eval queries.")
    parser.add_argument("--selected-docs", default="reports/phase5c3_table_expansion/selected_docs.csv")
    parser.add_argument("--association-audit", default="reports/phase5c3_table_expansion/association_audit/association_audit.csv")
    parser.add_argument("--phase4-approved-eval", default="reports/table_figure_retrieval_eval/phase4e3_eval_set_approved/eval_set.jsonl")
    parser.add_argument("--enhanced-chunks", default="/tmp/biorag_phase5c3_enhanced/chunks/chunks.jsonl")
    parser.add_argument("--baseline-chunks", default="/tmp/biorag_phase5c3_baseline/chunks.jsonl")
    parser.add_argument("--output-dir", default="reports/phase5c3_table_expansion")
    parser.add_argument("--table-content-count", type=int, default=70)
    parser.add_argument("--caption-count", type=int, default=40)
    parser.add_argument("--normal-count", type=int, default=40)
    parser.add_argument("--diagnostic-count", type=int, default=15)
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
        baseline_by_doc=baseline_by_doc,
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

    (output_dir / "eval_query_summary.md").write_text(summarize_queries(queries), encoding="utf-8")
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
    rows.sort(key=lambda r: (r["doc_id"], r.get("table_caption_block_id", ""), r.get("associated_block_id", "")))
    return rows


def chunk_block_ids(chunk: dict[str, Any] | None) -> list[str]:
    if not chunk:
        return []
    ids = chunk.get("source_block_ids") or chunk.get("block_ids") or []
    return [str(item) for item in ids if item]


def block_id_set(chunk: dict[str, Any]) -> set[str]:
    return set(chunk_block_ids(chunk))


def find_chunk_by_block(chunks: list[dict[str, Any]], block_id: str) -> dict[str, Any] | None:
    for chunk in chunks:
        if block_id and block_id in block_id_set(chunk):
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
    queries: list[dict[str, Any]] = []
    seen_enhanced_chunks: Counter[str] = Counter()
    per_doc: Counter[str] = Counter()
    ranked = sorted(
        assoc_rows,
        key=lambda row: (
            confidence_rank(row.get("association_confidence", "")),
            len(anchor_terms(row.get("associated_text_preview", ""), row.get("caption_text_preview", ""))),
            row.get("doc_id", ""),
        ),
        reverse=True,
    )
    for row in ranked:
        doc_id = row["doc_id"]
        anchors = anchor_terms(row.get("associated_text_preview", ""), row.get("caption_text_preview", ""))
        if len(anchors) < 2:
            continue
        enhanced = find_chunk_by_block(enhanced_by_doc[doc_id], row.get("associated_block_id", ""))
        if not enhanced:
            continue
        if seen_enhanced_chunks[enhanced["chunk_id"]] >= 2:
            continue
        if per_doc[doc_id] >= 4:
            continue
        baseline = find_caption_chunk(baseline_by_doc[doc_id], row.get("table_caption_block_id", ""), anchors)
        stable_ids = stable_ids_for(row.get("associated_block_id", ""), row.get("table_caption_block_id", ""))
        queries.append(add_target_aliases({
            "sample_id": f"p5c3_table_content_{len(queries) + 1:04d}",
            "query_type": "table_content",
            "query": table_content_query(anchors, row.get("associated_text_preview", ""), row.get("caption_text_preview", "")),
            "target_doc_id": doc_id,
            "target_caption_block_id": row.get("table_caption_block_id", ""),
            "target_associated_block_id": row.get("associated_block_id", ""),
            "stable_target_block_ids": stable_ids,
            "target_chunk_id_baseline": baseline.get("chunk_id", "") if baseline else "",
            "target_chunk_id_enhanced": enhanced.get("chunk_id", ""),
            "anchor_terms": anchors[:5],
            "expected_gain_source": "table_related_association",
            "include_in_main_denominator": True,
            "notes": f"association_confidence={row.get('association_confidence', '')}; rule_hits={row.get('rule_hits', '')}",
        }))
        seen_enhanced_chunks[enhanced["chunk_id"]] += 1
        per_doc[doc_id] += 1
        if len(queries) >= limit:
            break
    return queries


def confidence_rank(value: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(value, 0)


def table_content_query(anchors: list[str], associated_text: str, caption_text: str) -> str:
    joined = " ".join(anchors[:3])
    lower = f"{associated_text} {caption_text}".lower()
    if any(term in lower for term in ("strain", "plasmid", "primer", "genotype", "source", "orf")):
        return f"Which table lists {joined} information?"
    if any(term in lower for term in ("yield", "titer", "titre", "activity", "condition", "medium", "biomass", "rate")):
        return f"Which table reports {joined} measurements?"
    if any(term in lower for term in ("gene", "protein", "enzyme", "mutant", "homolog")):
        return f"Where are {joined} summarized in a table?"
    return f"Which table contains {joined}?"


def build_caption_queries(
    approved_path: Path,
    selected_doc_ids: set[str],
    baseline_by_id: dict[str, dict[str, Any]],
    baseline_by_doc: dict[str, list[dict[str, Any]]],
    enhanced_by_id: dict[str, dict[str, Any]],
    enhanced_by_doc: dict[str, list[dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    if not approved_path.exists():
        approved_items = []
    else:
        approved_items = list(iter_jsonl(approved_path))
    queries: list[dict[str, Any]] = []
    used_targets: set[str] = set()
    for sample in approved_items:
        if sample.get("target_doc_id") not in selected_doc_ids:
            continue
        if sample.get("sample_type") != "table" or sample.get("approved") is False:
            continue
        if sample.get("include_in_main_denominator") is False:
            continue
        doc_id = str(sample.get("target_doc_id", ""))
        original_target = str(sample.get("target_chunk_id", ""))
        anchors = [str(x) for x in sample.get("anchor_terms") or [] if str(x).strip()][:4]
        baseline_target = baseline_by_id.get(original_target)
        if not baseline_target:
            baseline_target = find_caption_chunk(baseline_by_doc[doc_id], "", anchors)
        caption_ids = table_caption_block_ids(baseline_target) if baseline_target else []
        caption_id = caption_ids[0] if caption_ids else ""
        enhanced_target = find_caption_chunk(enhanced_by_doc[doc_id], caption_id, anchors)
        if not enhanced_target and original_target in enhanced_by_id:
            enhanced_target = enhanced_by_id[original_target]
        queries.append(add_target_aliases({
            "sample_id": f"p5c3_caption_level_{len(queries) + 1:04d}",
            "query_type": "caption_level_table",
            "query": str(sample.get("query", "")),
            "target_doc_id": doc_id,
            "target_caption_block_id": caption_id,
            "target_associated_block_id": "",
            "stable_target_block_ids": caption_ids,
            "target_chunk_id_baseline": baseline_target.get("chunk_id", "") if baseline_target else original_target,
            "target_chunk_id_enhanced": enhanced_target.get("chunk_id", "") if enhanced_target else "",
            "anchor_terms": anchors,
            "expected_gain_source": "caption_level_regression_guard",
            "include_in_main_denominator": True,
            "notes": f"source_sample_id={sample.get('sample_id', '')}",
        }))
        if baseline_target:
            used_targets.add(baseline_target.get("chunk_id", ""))
        if len(queries) >= limit:
            break
    if len(queries) < limit:
        queries.extend(build_caption_fallback_queries(
            selected_doc_ids=selected_doc_ids,
            baseline_by_doc=baseline_by_doc,
            enhanced_by_doc=enhanced_by_doc,
            used_targets=used_targets,
            start_index=len(queries),
            limit=limit - len(queries),
        ))
    return queries


def build_caption_fallback_queries(
    *,
    selected_doc_ids: set[str],
    baseline_by_doc: dict[str, list[dict[str, Any]]],
    enhanced_by_doc: dict[str, list[dict[str, Any]]],
    used_targets: set[str],
    start_index: int,
    limit: int,
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    candidates = [
        chunk
        for doc_id in sorted(selected_doc_ids)
        for chunk in baseline_by_doc.get(doc_id, [])
        if chunk.get("contains_table_caption") and chunk.get("chunk_id") not in used_targets
    ]
    candidates.sort(key=lambda c: (c.get("doc_id", ""), c.get("chunk_index", 0), c.get("chunk_id", "")))
    for chunk in candidates:
        anchors = anchor_terms(chunk.get("text", ""), chunk.get("title", ""))
        if len(anchors) < 2:
            continue
        caption_ids = table_caption_block_ids(chunk)
        caption_id = caption_ids[0] if caption_ids else ""
        enhanced = find_caption_chunk(enhanced_by_doc[str(chunk.get("doc_id", ""))], caption_id, anchors)
        queries.append(add_target_aliases({
            "sample_id": f"p5c3_caption_level_{start_index + len(queries) + 1:04d}",
            "query_type": "caption_level_table",
            "query": f"Where are {' '.join(anchors[:3])} summarized in a table?",
            "target_doc_id": str(chunk.get("doc_id", "")),
            "target_caption_block_id": caption_id,
            "target_associated_block_id": "",
            "stable_target_block_ids": caption_ids,
            "target_chunk_id_baseline": chunk.get("chunk_id", ""),
            "target_chunk_id_enhanced": enhanced.get("chunk_id", "") if enhanced else "",
            "anchor_terms": anchors[:4],
            "expected_gain_source": "caption_level_regression_guard",
            "include_in_main_denominator": True,
            "notes": "fallback selected-doc table_caption chunk",
        }))
        if len(queries) >= limit:
            break
    return queries


def table_caption_block_ids(chunk: dict[str, Any] | None) -> list[str]:
    if not chunk:
        return []
    ids: list[str] = []
    for meta in chunk.get("source_block_metadata") or []:
        if isinstance(meta, dict) and meta.get("type") == "table_caption":
            value = meta.get("source_block_id") or meta.get("block_id")
            if value:
                ids.append(str(value))
    if not ids and chunk.get("contains_table_caption"):
        ids = chunk_block_ids(chunk)
    return ids


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
        if per_doc[doc_id] >= 1 and len(queries) < 32:
            continue
        anchors = anchor_terms(chunk.get("text", ""), chunk.get("title", ""))
        if len(anchors) < 2:
            continue
        enhanced = enhanced_by_id.get(chunk["chunk_id"]) or match_by_blocks(enhanced_by_doc[doc_id], chunk)
        stable_ids = non_table_block_ids(chunk)[:3] or chunk_block_ids(chunk)[:3]
        queries.append(add_target_aliases({
            "sample_id": f"p5c3_normal_control_{len(queries) + 1:04d}",
            "query_type": "normal_control",
            "query": normal_query(anchors, chunk),
            "target_doc_id": doc_id,
            "target_caption_block_id": "",
            "target_associated_block_id": "",
            "stable_target_block_ids": stable_ids,
            "target_chunk_id_baseline": chunk.get("chunk_id", ""),
            "target_chunk_id_enhanced": enhanced.get("chunk_id", "") if enhanced else "",
            "anchor_terms": anchors[:4],
            "expected_gain_source": "normal_control",
            "include_in_main_denominator": True,
            "notes": "paragraph control from selected docs",
        }))
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
    wanted = block_id_set(target)
    if not wanted:
        return None
    scored = []
    for chunk in chunks:
        overlap = len(wanted & block_id_set(chunk))
        if overlap:
            scored.append((overlap, chunk))
    scored.sort(key=lambda item: (-item[0], item[1].get("chunk_id", "")))
    return scored[0][1] if scored else None


def non_table_block_ids(chunk: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for meta in chunk.get("source_block_metadata") or []:
        if not isinstance(meta, dict):
            continue
        if meta.get("type") not in {"table_caption", "table_text", "figure_caption"}:
            block_id = meta.get("source_block_id") or meta.get("block_id")
            if block_id:
                ids.append(str(block_id))
    return ids


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
        queries.append(add_target_aliases({
            "sample_id": f"p5c3_diagnostic_{len(queries) + 1:04d}",
            "query_type": "diagnostic_only",
            "query": f"Where is {anchors[0]} mentioned?",
            "target_doc_id": row["doc_id"],
            "target_caption_block_id": row.get("table_caption_block_id", ""),
            "target_associated_block_id": row.get("associated_block_id", ""),
            "stable_target_block_ids": stable_ids_for(row.get("associated_block_id", ""), row.get("table_caption_block_id", "")),
            "target_chunk_id_baseline": "",
            "target_chunk_id_enhanced": enhanced.get("chunk_id", ""),
            "anchor_terms": anchors[:3],
            "expected_gain_source": "table_related_association",
            "include_in_main_denominator": False,
            "notes": "ambiguous diagnostic probe; excluded from main denominator",
        }))
        if len(queries) >= limit:
            break
    return queries


def stable_ids_for(*values: str) -> list[str]:
    return [str(value) for value in values if value]


def add_target_aliases(query: dict[str, Any]) -> dict[str, Any]:
    query["target_chunk_id_baseline_candidate"] = query.get("target_chunk_id_baseline", "")
    query["target_chunk_id_enhanced_candidate"] = query.get("target_chunk_id_enhanced", "")
    return query


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
        "# Phase 5C-3 Eval Query Summary",
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
        "- Queries were generated deterministically from selected 5C-3 artifacts.",
        "- table_content targets accepted table_related associated blocks and uses source block ids for corrected metrics.",
        "- caption_level_table checks Phase 4 approved table-caption behavior.",
        "- normal_control checks paragraph retrieval and table-related takeover risk.",
        "- diagnostic_only probes are excluded from the main denominator.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
