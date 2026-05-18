#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HELPER_PATH = Path(__file__).with_name("prepare_phase5c3_table_retrieval_eval.py")
spec = importlib.util.spec_from_file_location("phase5c3_prepare", HELPER_PATH)
phase5c3 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(phase5c3)

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._'+/\-()%]{1,}")
STOP = set(getattr(phase5c3, "STOP", set())) | {
    "fig", "supplementary", "data", "shown", "using", "based", "between",
    "including", "respectively", "significant", "however", "therefore",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic Phase 5C-5 full retrieval eval queries.")
    parser.add_argument("--baseline-chunks", default="/tmp/biorag_phase4d_compact_chunks/chunks.jsonl")
    parser.add_argument("--enhanced-chunks", default="/tmp/biorag_phase5c4_full_enhanced/chunks/chunks.jsonl")
    parser.add_argument("--chunk-ab-stats", default="reports/phase5c4_full_preflight/full_chunk_ab_stats.json")
    parser.add_argument("--association-audit", default="reports/phase5c4_full_preflight/pilot_association_audit.csv")
    parser.add_argument("--association-summary", default="reports/phase5c4_full_preflight/dryrun_association_summary.json")
    parser.add_argument("--doc-level-stats", default="reports/phase5c4_full_preflight/dryrun_doc_level_stats.csv")
    parser.add_argument("--truncation-risk-chunks", default="reports/phase5c4_full_preflight/truncation_risk_chunks.csv")
    parser.add_argument("--stable-target-summary", default="reports/phase5c2_table_retrieval_ab/stable_target_mapping/summary.md")
    parser.add_argument("--phase5c3-summary", default="reports/phase5c3_table_expansion/retrieval_ab/summary.md")
    parser.add_argument("--phase4-approved-eval", default="reports/table_figure_retrieval_eval/phase4e3_eval_set_approved/eval_set.jsonl")
    parser.add_argument("--output-dir", default="reports/phase5c5_full_retrieval_ab")
    parser.add_argument("--table-content-count", type=int, default=130)
    parser.add_argument("--caption-count", type=int, default=70)
    parser.add_argument("--figure-count", type=int, default=50)
    parser.add_argument("--normal-count", type=int, default=90)
    parser.add_argument("--risk-count", type=int, default=45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_chunks = list(iter_jsonl(Path(args.baseline_chunks)))
    enhanced_chunks = list(iter_jsonl(Path(args.enhanced_chunks)))
    baseline_by_doc = group_by_doc(baseline_chunks)
    enhanced_by_doc = group_by_doc(enhanced_chunks)
    baseline_by_id = {c["chunk_id"]: c for c in baseline_chunks}
    enhanced_by_id = {c["chunk_id"]: c for c in enhanced_chunks}
    assoc_rows = load_associations(Path(args.association_audit))
    doc_stats = load_doc_stats(Path(args.doc_level_stats))
    truncation_docs = load_truncation_docs(Path(args.truncation_risk_chunks))
    approved = load_approved(Path(args.phase4_approved_eval))

    queries: list[dict[str, Any]] = []
    queries.extend(build_table_content_queries(assoc_rows, baseline_by_doc, enhanced_by_doc, args.table_content_count))
    queries.extend(build_caption_queries(approved, baseline_chunks, baseline_by_id, baseline_by_doc, enhanced_by_id, enhanced_by_doc, args.caption_count))
    queries.extend(build_figure_queries(approved, baseline_chunks, baseline_by_id, baseline_by_doc, enhanced_by_id, enhanced_by_doc, args.figure_count))
    queries.extend(build_normal_queries(approved, baseline_chunks, baseline_by_id, baseline_by_doc, enhanced_by_id, enhanced_by_doc, args.normal_count))
    queries.extend(build_risk_queries(assoc_rows, doc_stats, truncation_docs, baseline_chunks, baseline_by_doc, enhanced_by_doc, args.risk_count))

    out_jsonl = out_dir / "eval_queries.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for item in queries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    (out_dir / "eval_query_summary.md").write_text(
        summarize_queries(queries, args, baseline_chunks, enhanced_chunks),
        encoding="utf-8",
    )
    print(f"Wrote {out_jsonl}")
    print(f"Wrote {out_dir / 'eval_query_summary.md'}")


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


def load_associations(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("accepted_or_rejected") == "accepted":
                rows.append(row)
    rows.sort(key=lambda r: (r.get("doc_id", ""), r.get("table_caption_block_id", ""), r.get("associated_block_id", "")))
    return rows


def load_doc_stats(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["doc_id"]: row for row in csv.DictReader(f) if row.get("doc_id")}


def load_truncation_docs(path: Path) -> set[str]:
    docs: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("bge_gt_4096") == "True":
                docs.add(str(row.get("doc_id", "")))
    return docs


def load_approved(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path)) if path.exists() else []


def build_table_content_queries(
    assoc_rows: list[dict[str, str]],
    baseline_by_doc: dict[str, list[dict[str, Any]]],
    enhanced_by_doc: dict[str, list[dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        assoc_rows,
        key=lambda row: (
            confidence_rank(row.get("association_confidence", "")),
            len(anchor_terms(row.get("associated_text_preview", ""), row.get("caption_text_preview", ""))),
            -int(row.get("block_distance") or 999),
            row.get("doc_id", ""),
        ),
        reverse=True,
    )
    queries: list[dict[str, Any]] = []
    per_doc: Counter[str] = Counter()
    per_enhanced_chunk: Counter[str] = Counter()
    for row in ranked:
        doc_id = row["doc_id"]
        anchors = anchor_terms(row.get("associated_text_preview", ""), row.get("caption_text_preview", ""))
        if len(anchors) < 2 or per_doc[doc_id] >= 3:
            continue
        enhanced = find_chunk_by_block(enhanced_by_doc.get(doc_id, []), row.get("associated_block_id", ""))
        if not enhanced or per_enhanced_chunk[enhanced["chunk_id"]] >= 2:
            continue
        baseline = find_caption_chunk(baseline_by_doc.get(doc_id, []), row.get("table_caption_block_id", ""), anchors)
        queries.append(add_aliases({
            "sample_id": f"p5c5_table_content_{len(queries) + 1:04d}",
            "query_type": "table_content",
            "query": table_content_query(anchors, row.get("associated_text_preview", ""), row.get("caption_text_preview", "")),
            "target_doc_id": doc_id,
            "stable_target_block_ids": stable_ids_for(row.get("associated_block_id", ""), row.get("table_caption_block_id", "")),
            "target_caption_block_id": row.get("table_caption_block_id", ""),
            "target_associated_block_id": row.get("associated_block_id", ""),
            "target_chunk_id_baseline": baseline.get("chunk_id", "") if baseline else "",
            "target_chunk_id_enhanced": enhanced.get("chunk_id", ""),
            "anchor_terms": anchors[:5],
            "include_in_main_denominator": True,
            "risk_tags": risk_tags_for_assoc(row),
            "expected_gain_source": "table_related_association",
            "notes": f"confidence={row.get('association_confidence', '')}; rule_hits={row.get('rule_hits', '')}",
        }))
        per_doc[doc_id] += 1
        per_enhanced_chunk[enhanced["chunk_id"]] += 1
        if len(queries) >= limit:
            break
    return queries


def build_caption_queries(
    approved: list[dict[str, Any]],
    baseline_chunks: list[dict[str, Any]],
    baseline_by_id: dict[str, dict[str, Any]],
    baseline_by_doc: dict[str, list[dict[str, Any]]],
    enhanced_by_id: dict[str, dict[str, Any]],
    enhanced_by_doc: dict[str, list[dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    used: set[str] = set()
    for sample in approved:
        if sample.get("sample_type") != "table" or sample.get("approved") is False:
            continue
        doc_id = str(sample.get("target_doc_id", ""))
        baseline = baseline_by_id.get(str(sample.get("target_chunk_id", "")))
        anchors = [str(x) for x in sample.get("anchor_terms") or [] if str(x).strip()][:5]
        if not baseline:
            baseline = find_caption_chunk(baseline_by_doc.get(doc_id, []), "", anchors)
        caption_ids = table_caption_block_ids(baseline)
        enhanced = find_caption_chunk(enhanced_by_doc.get(doc_id, []), caption_ids[0] if caption_ids else "", anchors)
        if not enhanced:
            enhanced = enhanced_by_id.get(str(sample.get("target_chunk_id", "")))
        queries.append(add_aliases({
            "sample_id": f"p5c5_caption_level_table_{len(queries) + 1:04d}",
            "query_type": "caption_level_table",
            "query": str(sample.get("query", "")),
            "target_doc_id": doc_id,
            "stable_target_block_ids": caption_ids,
            "target_caption_block_id": caption_ids[0] if caption_ids else "",
            "target_associated_block_id": "",
            "target_chunk_id_baseline": baseline.get("chunk_id", "") if baseline else str(sample.get("target_chunk_id", "")),
            "target_chunk_id_enhanced": enhanced.get("chunk_id", "") if enhanced else "",
            "anchor_terms": anchors,
            "include_in_main_denominator": True,
            "risk_tags": [],
            "expected_gain_source": "caption_level_table_guard",
            "notes": f"phase4_sample_id={sample.get('sample_id', '')}",
        }))
        if baseline:
            used.add(baseline["chunk_id"])
        if len(queries) >= limit:
            return queries

    candidates = [c for c in baseline_chunks if c.get("contains_table_caption") and c.get("chunk_id") not in used]
    candidates.sort(key=lambda c: (c.get("doc_id", ""), c.get("chunk_index", 0), c.get("chunk_id", "")))
    for chunk in candidates:
        anchors = anchor_terms(chunk.get("text", ""), chunk.get("title", ""))
        if len(anchors) < 2:
            continue
        doc_id = str(chunk.get("doc_id", ""))
        caption_ids = table_caption_block_ids(chunk)
        enhanced = find_caption_chunk(enhanced_by_doc.get(doc_id, []), caption_ids[0] if caption_ids else "", anchors)
        queries.append(add_aliases({
            "sample_id": f"p5c5_caption_level_table_{len(queries) + 1:04d}",
            "query_type": "caption_level_table",
            "query": f"Which table reports {' '.join(anchors[:3])}?",
            "target_doc_id": doc_id,
            "stable_target_block_ids": caption_ids,
            "target_caption_block_id": caption_ids[0] if caption_ids else "",
            "target_associated_block_id": "",
            "target_chunk_id_baseline": chunk.get("chunk_id", ""),
            "target_chunk_id_enhanced": enhanced.get("chunk_id", "") if enhanced else "",
            "anchor_terms": anchors[:5],
            "include_in_main_denominator": True,
            "risk_tags": [],
            "expected_gain_source": "caption_level_table_guard",
            "notes": "full-corpus table caption fallback",
        }))
        if len(queries) >= limit:
            break
    return queries


def build_figure_queries(
    approved: list[dict[str, Any]],
    baseline_chunks: list[dict[str, Any]],
    baseline_by_id: dict[str, dict[str, Any]],
    baseline_by_doc: dict[str, list[dict[str, Any]]],
    enhanced_by_id: dict[str, dict[str, Any]],
    enhanced_by_doc: dict[str, list[dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    used: set[str] = set()
    for sample in approved:
        if sample.get("sample_type") != "figure" or sample.get("approved") is False:
            continue
        doc_id = str(sample.get("target_doc_id", ""))
        baseline = baseline_by_id.get(str(sample.get("target_chunk_id", "")))
        anchors = [str(x) for x in sample.get("anchor_terms") or [] if str(x).strip()][:5]
        if not baseline:
            baseline = find_by_blocks_or_anchors(baseline_by_doc.get(doc_id, []), [], anchors, "figure_caption")
        stable_ids = figure_caption_block_ids(baseline)
        enhanced = find_by_blocks_or_anchors(enhanced_by_doc.get(doc_id, []), stable_ids, anchors, "figure_caption")
        if not enhanced:
            enhanced = enhanced_by_id.get(str(sample.get("target_chunk_id", "")))
        queries.append(add_aliases({
            "sample_id": f"p5c5_figure_caption_{len(queries) + 1:04d}",
            "query_type": "figure_caption",
            "query": str(sample.get("query", "")),
            "target_doc_id": doc_id,
            "stable_target_block_ids": stable_ids,
            "target_caption_block_id": stable_ids[0] if stable_ids else "",
            "target_associated_block_id": "",
            "target_chunk_id_baseline": baseline.get("chunk_id", "") if baseline else str(sample.get("target_chunk_id", "")),
            "target_chunk_id_enhanced": enhanced.get("chunk_id", "") if enhanced else "",
            "anchor_terms": anchors,
            "include_in_main_denominator": True,
            "risk_tags": [],
            "expected_gain_source": "figure_caption_guard",
            "notes": f"phase4_sample_id={sample.get('sample_id', '')}",
        }))
        if baseline:
            used.add(baseline["chunk_id"])
        if len(queries) >= limit:
            return queries

    candidates = [c for c in baseline_chunks if c.get("contains_figure_caption") and c.get("chunk_id") not in used]
    candidates.sort(key=lambda c: (c.get("doc_id", ""), c.get("chunk_index", 0), c.get("chunk_id", "")))
    for chunk in candidates:
        anchors = anchor_terms(chunk.get("text", ""), chunk.get("title", ""))
        if len(anchors) < 2:
            continue
        doc_id = str(chunk.get("doc_id", ""))
        stable_ids = figure_caption_block_ids(chunk)
        enhanced = find_by_blocks_or_anchors(enhanced_by_doc.get(doc_id, []), stable_ids, anchors, "figure_caption")
        queries.append(add_aliases({
            "sample_id": f"p5c5_figure_caption_{len(queries) + 1:04d}",
            "query_type": "figure_caption",
            "query": f"Which figure shows {' '.join(anchors[:3])}?",
            "target_doc_id": doc_id,
            "stable_target_block_ids": stable_ids,
            "target_caption_block_id": stable_ids[0] if stable_ids else "",
            "target_associated_block_id": "",
            "target_chunk_id_baseline": chunk.get("chunk_id", ""),
            "target_chunk_id_enhanced": enhanced.get("chunk_id", "") if enhanced else "",
            "anchor_terms": anchors[:5],
            "include_in_main_denominator": True,
            "risk_tags": [],
            "expected_gain_source": "figure_caption_guard",
            "notes": "full-corpus figure caption fallback",
        }))
        if len(queries) >= limit:
            break
    return queries


def build_normal_queries(
    approved: list[dict[str, Any]],
    baseline_chunks: list[dict[str, Any]],
    baseline_by_id: dict[str, dict[str, Any]],
    baseline_by_doc: dict[str, list[dict[str, Any]]],
    enhanced_by_id: dict[str, dict[str, Any]],
    enhanced_by_doc: dict[str, list[dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    used: set[str] = set()
    for sample in approved:
        if sample.get("sample_type") != "normal" or sample.get("approved") is False:
            continue
        doc_id = str(sample.get("target_doc_id", ""))
        baseline = baseline_by_id.get(str(sample.get("target_chunk_id", "")))
        anchors = [str(x) for x in sample.get("anchor_terms") or [] if str(x).strip()][:5]
        if not baseline:
            baseline = find_by_blocks_or_anchors(baseline_by_doc.get(doc_id, []), [], anchors, "normal")
        stable_ids = non_table_block_ids(baseline)[:3] or chunk_block_ids(baseline)[:3]
        enhanced = enhanced_by_id.get(str(sample.get("target_chunk_id", ""))) or match_by_blocks(enhanced_by_doc.get(doc_id, []), baseline)
        queries.append(add_aliases({
            "sample_id": f"p5c5_normal_control_{len(queries) + 1:04d}",
            "query_type": "normal_control",
            "query": str(sample.get("query", "")),
            "target_doc_id": doc_id,
            "stable_target_block_ids": stable_ids,
            "target_caption_block_id": "",
            "target_associated_block_id": "",
            "target_chunk_id_baseline": baseline.get("chunk_id", "") if baseline else str(sample.get("target_chunk_id", "")),
            "target_chunk_id_enhanced": enhanced.get("chunk_id", "") if enhanced else "",
            "anchor_terms": anchors,
            "include_in_main_denominator": True,
            "risk_tags": [],
            "expected_gain_source": "normal_control",
            "notes": f"phase4_sample_id={sample.get('sample_id', '')}",
        }))
        if baseline:
            used.add(baseline["chunk_id"])
        if len(queries) >= limit:
            return queries

    candidates = [c for c in baseline_chunks if is_normal_paragraph(c) and c.get("chunk_id") not in used and len(c.get("text", "")) >= 250]
    candidates.sort(key=lambda c: (c.get("doc_id", ""), c.get("chunk_index", 0), c.get("chunk_id", "")))
    per_doc: Counter[str] = Counter()
    for chunk in candidates:
        doc_id = str(chunk.get("doc_id", ""))
        if per_doc[doc_id] >= 1 and len(queries) < limit - 10:
            continue
        anchors = anchor_terms(chunk.get("text", ""), chunk.get("title", ""))
        if len(anchors) < 2:
            continue
        enhanced = match_by_blocks(enhanced_by_doc.get(doc_id, []), chunk)
        stable_ids = non_table_block_ids(chunk)[:3] or chunk_block_ids(chunk)[:3]
        queries.append(add_aliases({
            "sample_id": f"p5c5_normal_control_{len(queries) + 1:04d}",
            "query_type": "normal_control",
            "query": normal_query(anchors, chunk),
            "target_doc_id": doc_id,
            "stable_target_block_ids": stable_ids,
            "target_caption_block_id": "",
            "target_associated_block_id": "",
            "target_chunk_id_baseline": chunk.get("chunk_id", ""),
            "target_chunk_id_enhanced": enhanced.get("chunk_id", "") if enhanced else "",
            "anchor_terms": anchors[:5],
            "include_in_main_denominator": True,
            "risk_tags": [],
            "expected_gain_source": "normal_control",
            "notes": "full-corpus paragraph fallback",
        }))
        per_doc[doc_id] += 1
        if len(queries) >= limit:
            break
    return queries


def build_risk_queries(
    assoc_rows: list[dict[str, str]],
    doc_stats: dict[str, dict[str, str]],
    truncation_docs: set[str],
    baseline_chunks: list[dict[str, Any]],
    baseline_by_doc: dict[str, list[dict[str, Any]]],
    enhanced_by_doc: dict[str, list[dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    used_keys: set[tuple[str, str]] = set()

    buckets = [
        ("low_confidence_association", lambda r: r.get("association_confidence") == "low"),
        ("uncertain_case_doc", lambda r: int(doc_stats.get(r.get("doc_id", ""), {}).get("uncertain_cases") or 0) > 0),
        ("high_association_doc", lambda r: int(doc_stats.get(r.get("doc_id", ""), {}).get("accepted_associations") or 0) >= 8),
        ("table_related_chunk", lambda r: True),
    ]
    for special_doc in ("doc_0377", "doc_0442"):
        for row in assoc_rows:
            if row.get("doc_id") != special_doc:
                continue
            item = risk_query_from_assoc(row, special_doc, baseline_by_doc, enhanced_by_doc, len(queries) + 1)
            if not item:
                continue
            key = (item["target_doc_id"], item["target_associated_block_id"])
            if key in used_keys:
                continue
            used_keys.add(key)
            item["risk_tags"] = sorted(set(item.get("risk_tags", []) + [special_doc, "special_suspicious_doc"]))
            queries.append(item)
            if len([q for q in queries if special_doc in q.get("risk_tags", [])]) >= 4:
                break
    for tag, pred in buckets:
        for row in assoc_rows:
            if not pred(row):
                continue
            item = risk_query_from_assoc(row, tag, baseline_by_doc, enhanced_by_doc, len(queries) + 1)
            if not item:
                continue
            key = (item["target_doc_id"], item["target_associated_block_id"])
            if key in used_keys:
                continue
            used_keys.add(key)
            queries.append(item)
            if len([q for q in queries if tag in q.get("risk_tags", [])]) >= 8:
                break
        if len(queries) >= limit:
            return queries[:limit]

    trunc_candidates = [
        c for c in baseline_chunks
        if c.get("doc_id") in truncation_docs and is_normal_paragraph(c)
    ]
    trunc_candidates.sort(key=lambda c: (c.get("doc_id", ""), c.get("chunk_index", 0), c.get("chunk_id", "")))
    for chunk in trunc_candidates:
        anchors = anchor_terms(chunk.get("text", ""), chunk.get("title", ""))
        if len(anchors) < 2:
            continue
        doc_id = str(chunk.get("doc_id", ""))
        enhanced = match_by_blocks(enhanced_by_doc.get(doc_id, []), chunk)
        queries.append(add_aliases({
            "sample_id": f"p5c5_risk_slice_{len(queries) + 1:04d}",
            "query_type": "risk_slice",
            "query": normal_query(anchors, chunk),
            "target_doc_id": doc_id,
            "stable_target_block_ids": non_table_block_ids(chunk)[:3] or chunk_block_ids(chunk)[:3],
            "target_caption_block_id": "",
            "target_associated_block_id": "",
            "target_chunk_id_baseline": chunk.get("chunk_id", ""),
            "target_chunk_id_enhanced": enhanced.get("chunk_id", "") if enhanced else "",
            "anchor_terms": anchors[:5],
            "include_in_main_denominator": False,
            "risk_tags": ["tokenizer_truncation_risk_doc"],
            "expected_gain_source": "risk_diagnostic",
            "notes": "BGE tokenizer truncation-risk doc diagnostic",
        }))
        if len(queries) >= limit:
            break
    return queries[:limit]


def risk_query_from_assoc(
    row: dict[str, str],
    tag: str,
    baseline_by_doc: dict[str, list[dict[str, Any]]],
    enhanced_by_doc: dict[str, list[dict[str, Any]]],
    index: int,
) -> dict[str, Any] | None:
    doc_id = row["doc_id"]
    anchors = anchor_terms(row.get("associated_text_preview", ""), row.get("caption_text_preview", ""))
    if not anchors:
        return None
    enhanced = find_chunk_by_block(enhanced_by_doc.get(doc_id, []), row.get("associated_block_id", ""))
    if not enhanced:
        return None
    baseline = find_caption_chunk(baseline_by_doc.get(doc_id, []), row.get("table_caption_block_id", ""), anchors)
    tags = sorted(set(risk_tags_for_assoc(row) + [tag, "table_related_chunk"]))
    return add_aliases({
        "sample_id": f"p5c5_risk_slice_{index:04d}",
        "query_type": "risk_slice",
        "query": f"Where is {' '.join(anchors[:3])} reported?",
        "target_doc_id": doc_id,
        "stable_target_block_ids": stable_ids_for(row.get("associated_block_id", ""), row.get("table_caption_block_id", "")),
        "target_caption_block_id": row.get("table_caption_block_id", ""),
        "target_associated_block_id": row.get("associated_block_id", ""),
        "target_chunk_id_baseline": baseline.get("chunk_id", "") if baseline else "",
        "target_chunk_id_enhanced": enhanced.get("chunk_id", ""),
        "anchor_terms": anchors[:5],
        "include_in_main_denominator": False,
        "risk_tags": tags,
        "expected_gain_source": "risk_diagnostic_table_related",
        "notes": f"confidence={row.get('association_confidence', '')}; rule_hits={row.get('rule_hits', '')}",
    })


def confidence_rank(value: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(value, 0)


def table_content_query(anchors: list[str], associated_text: str, caption_text: str) -> str:
    joined = " ".join(anchors[:3])
    lower = f"{associated_text} {caption_text}".lower()
    if any(t in lower for t in ("strain", "plasmid", "primer", "genotype", "source", "orf")):
        return f"Which table lists {joined} information?"
    if any(t in lower for t in ("yield", "titer", "titre", "activity", "condition", "medium", "biomass", "rate")):
        return f"Which table reports {joined} measurements?"
    if any(t in lower for t in ("gene", "protein", "enzyme", "mutant", "homolog")):
        return f"Where are {joined} summarized in a table?"
    return f"Which table contains {joined}?"


def normal_query(anchors: list[str], chunk: dict[str, Any]) -> str:
    section = str(chunk.get("section") or "the paper")
    return f"What does {section} report about {' '.join(anchors[:3])}?"


def find_chunk_by_block(chunks: list[dict[str, Any]], block_id: str) -> dict[str, Any] | None:
    if not block_id:
        return None
    for chunk in chunks:
        if block_id in set(chunk_block_ids(chunk)):
            return chunk
    return None


def find_caption_chunk(chunks: list[dict[str, Any]], caption_block_id: str, anchors: list[str]) -> dict[str, Any] | None:
    exact = find_chunk_by_block(chunks, caption_block_id)
    if exact:
        return exact
    return find_by_blocks_or_anchors(chunks, [], anchors, "table_caption")


def find_by_blocks_or_anchors(chunks: list[dict[str, Any]], block_ids: list[str], anchors: list[str], kind: str) -> dict[str, Any] | None:
    for block_id in block_ids:
        exact = find_chunk_by_block(chunks, block_id)
        if exact:
            return exact
    anchor_l = [a.lower() for a in anchors if a]
    scored: list[tuple[int, int, str, dict[str, Any]]] = []
    for chunk in chunks:
        if kind == "table_caption" and not chunk.get("contains_table_caption"):
            continue
        if kind == "figure_caption" and not chunk.get("contains_figure_caption"):
            continue
        if kind == "normal" and not is_normal_paragraph(chunk):
            continue
        text = f"{chunk.get('text', '')}\n{chunk.get('retrieval_text', '')}".lower()
        score = sum(1 for anchor in anchor_l if anchor.lower() in text)
        if score:
            scored.append((score, -len(text), chunk.get("chunk_id", ""), chunk))
    scored.sort(reverse=True)
    return scored[0][3] if scored else None


def match_by_blocks(chunks: list[dict[str, Any]], target: dict[str, Any] | None) -> dict[str, Any] | None:
    if not target:
        return None
    wanted = set(chunk_block_ids(target))
    scored: list[tuple[int, str, dict[str, Any]]] = []
    for chunk in chunks:
        overlap = len(wanted & set(chunk_block_ids(chunk)))
        if overlap:
            scored.append((overlap, chunk.get("chunk_id", ""), chunk))
    scored.sort(reverse=True)
    return scored[0][2] if scored else None


def chunk_block_ids(chunk: dict[str, Any] | None) -> list[str]:
    if not chunk:
        return []
    return [str(x) for x in (chunk.get("source_block_ids") or chunk.get("block_ids") or []) if x]


def typed_block_ids(chunk: dict[str, Any] | None, block_type: str) -> list[str]:
    if not chunk:
        return []
    ids: list[str] = []
    for meta in chunk.get("source_block_metadata") or []:
        if isinstance(meta, dict) and meta.get("type") == block_type:
            block_id = meta.get("source_block_id") or meta.get("block_id")
            if block_id:
                ids.append(str(block_id))
    if not ids:
        if block_type == "table_caption" and chunk.get("contains_table_caption"):
            ids = chunk_block_ids(chunk)
        elif block_type == "figure_caption" and chunk.get("contains_figure_caption"):
            ids = chunk_block_ids(chunk)
    return ids


def table_caption_block_ids(chunk: dict[str, Any] | None) -> list[str]:
    return typed_block_ids(chunk, "table_caption")


def figure_caption_block_ids(chunk: dict[str, Any] | None) -> list[str]:
    return typed_block_ids(chunk, "figure_caption")


def non_table_block_ids(chunk: dict[str, Any] | None) -> list[str]:
    if not chunk:
        return []
    ids: list[str] = []
    for meta in chunk.get("source_block_metadata") or []:
        if not isinstance(meta, dict):
            continue
        if meta.get("type") not in {"table_caption", "table_text", "figure_caption"}:
            block_id = meta.get("source_block_id") or meta.get("block_id")
            if block_id:
                ids.append(str(block_id))
    return ids


def is_normal_paragraph(chunk: dict[str, Any]) -> bool:
    if chunk.get("contains_table_text") or chunk.get("contains_table_caption") or chunk.get("contains_figure_caption"):
        return False
    if chunk.get("contains_references") or chunk.get("contains_metadata") or chunk.get("contains_noise"):
        return False
    types = set(chunk.get("evidence_types") or chunk.get("block_types") or [])
    return "paragraph" in types


def anchor_terms(primary: str, secondary: str = "") -> list[str]:
    text = f"{primary} {secondary}"
    terms: list[str] = []
    seen: set[str] = set()
    for raw in TOKEN_RE.findall(text):
        token = raw.strip(".,;:()[]{}")
        key = token.lower()
        if key in STOP or len(token) < 3:
            continue
        informative = any(ch.isdigit() for ch in token) or any(ch.isupper() for ch in token) or "-" in token or "%" in token or len(token) >= 6
        if not informative:
            continue
        if key not in seen:
            seen.add(key)
            terms.append(token)
        if len(terms) >= 6:
            break
    return terms


def risk_tags_for_assoc(row: dict[str, str]) -> list[str]:
    tags: list[str] = []
    conf = row.get("association_confidence", "")
    if conf:
        tags.append(f"{conf}_confidence_association")
    if row.get("doc_id") in {"doc_0377", "doc_0442"}:
        tags.append(row.get("doc_id", ""))
    return tags


def stable_ids_for(*values: str) -> list[str]:
    return [str(v) for v in values if v]


def add_aliases(query: dict[str, Any]) -> dict[str, Any]:
    query["target_chunk_id_baseline_candidate"] = query.get("target_chunk_id_baseline", "")
    query["target_chunk_id_enhanced_candidate"] = query.get("target_chunk_id_enhanced", "")
    return query


def summarize_queries(
    queries: list[dict[str, Any]],
    args: argparse.Namespace,
    baseline_chunks: list[dict[str, Any]],
    enhanced_chunks: list[dict[str, Any]],
) -> str:
    by_type = Counter(q["query_type"] for q in queries)
    main_by_type = Counter(q["query_type"] for q in queries if q.get("include_in_main_denominator"))
    risk_tags = Counter(tag for q in queries for tag in q.get("risk_tags", []))
    lines = [
        "# Phase 5C-5 Full Retrieval Eval Query Summary",
        "",
        f"- baseline chunks: {len(baseline_chunks)}",
        f"- enhanced chunks: {len(enhanced_chunks)}",
        f"- total queries: {len(queries)}",
        f"- main denominator queries: {sum(main_by_type.values())}",
        f"- qwen_called: false",
        f"- deterministic_generation: true",
        f"- stable_target_mapping_required: true",
        f"- stable target summary exists: {str(Path(args.stable_target_summary).exists()).lower()}",
        f"- phase5c3 retrieval summary exists: {str(Path(args.phase5c3_summary).exists()).lower()}",
        "",
        "## By Query Type",
        "",
    ]
    for qtype in sorted(by_type):
        lines.append(f"- {qtype}: {by_type[qtype]} total, {main_by_type[qtype]} main")
    lines.extend(["", "## Risk Tags", ""])
    for tag, count in risk_tags.most_common():
        lines.append(f"- {tag}: {count}")
    lines.extend([
        "",
        "## Notes",
        "",
        "- table_content targets enhanced table_related associated blocks and baseline caption-level candidates.",
        "- caption_level_table, figure_caption, and normal_control protect existing retrieval behavior.",
        "- risk_slice is excluded from the main denominator and covers low confidence, uncertain, high-association, suspicious docs, tokenizer-risk docs, and table_related chunks.",
        "- Cross-version exact matching must use stable source block ids rather than chunk_id equality.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
