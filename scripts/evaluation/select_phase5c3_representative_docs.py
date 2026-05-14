#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TABLE_KEYWORDS_RE = re.compile(
    r"\b(?:strain|plasmid|genotype|source|primer|sequence|yield|titer|titre|"
    r"condition|medium|activity|concentration|temperature|time|sample|gene|"
    r"protein|substrate|product|host|vector|enzyme|mutation|variant|species|"
    r"isolate|accession|name|permutant|unit|kd|ra|od600|length|bp|orf)\b",
    re.I,
)
CJK_TABLE_KEYWORDS_RE = re.compile(
    r"(?:表|序号|菌株|质粒|引物|浓度|条件|产量|方式|优势|劣势|反应式|能量|消耗|"
    r"甲醇|甲烷|乙烯|乙醇|基因|培养基|活性|来源|编号)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select representative Phase 5C-3 expansion docs.")
    parser.add_argument("--phase5c1-selected", default="reports/phase5_table_audit/selected_docs.csv")
    parser.add_argument("--table-flow-audit", default="reports/phase5_table_audit/table_flow_audit.csv")
    parser.add_argument("--phase5c1-association-audit", default="reports/phase5c1_table_preservation/association_audit.csv")
    parser.add_argument("--phase5c2-eval-queries", default="reports/phase5c2_table_retrieval_ab/eval_queries.jsonl")
    parser.add_argument("--phase5c2-target-mapping-summary", default="reports/phase5c2_table_retrieval_ab/stable_target_mapping/summary.md")
    parser.add_argument("--phase4-approved-eval", default="reports/table_figure_retrieval_eval/phase4e3_eval_set_approved/eval_set.jsonl")
    parser.add_argument("--phase4-manual-results", default="reports/table_figure_retrieval_eval/phase4e3_manual_eval/retrieval_results.json")
    parser.add_argument("--parsed-clean-dir", default="data/paper_round1/parsed_clean")
    parser.add_argument("--baseline-chunks", default="/tmp/biorag_phase4d_compact_chunks/chunks.jsonl")
    parser.add_argument("--output-dir", default="reports/phase5c3_table_expansion")
    parser.add_argument("--target-doc-count", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase5c1_rows = read_csv(Path(args.phase5c1_selected))
    phase5c1_ids = [row["doc_id"] for row in phase5c1_rows if row.get("doc_id")]
    phase5c1_set = set(phase5c1_ids)
    table_flow = {row["doc_id"]: row for row in read_csv(Path(args.table_flow_audit)) if row.get("doc_id")}
    phase5c1_assoc_counts = accepted_assoc_counts(Path(args.phase5c1_association_audit))
    phase5c2_table_docs = eval_docs(Path(args.phase5c2_eval_queries), query_type="table_content")
    phase4_table_docs = phase4_approved_table_docs(Path(args.phase4_approved_eval))
    edge_docs = edge_case_docs(Path(args.phase4_manual_results)) | phase5c2_table_docs
    chunks_by_doc = baseline_chunk_stats(Path(args.baseline_chunks))
    doc_stats = collect_doc_stats(Path(args.parsed_clean_dir), table_flow, chunks_by_doc, phase5c1_assoc_counts)

    selected: dict[str, dict[str, Any]] = {}
    bucket_counts: Counter[str] = Counter()

    def add_doc(doc_id: str, bucket: str, reason: str) -> None:
        if doc_id not in doc_stats:
            return
        row = selected.get(doc_id)
        if row is None:
            row = dict(doc_stats[doc_id])
            row["selection_bucket"] = bucket
            row["reason_selected"] = reason
            row["from_phase5c1_pilot"] = str(doc_id in phase5c1_set).lower()
            selected[doc_id] = row
            bucket_counts[bucket] += 1
        else:
            reasons = set(str(row["reason_selected"]).split("; "))
            if reason not in reasons:
                row["reason_selected"] = f"{row['reason_selected']}; {reason}"

    for doc_id in phase5c1_ids:
        add_doc(doc_id, "phase5c1_carryover", "Phase 5C-1 pilot carryover")
    for doc_id in sorted(phase4_table_docs | phase5c2_table_docs):
        add_doc(doc_id, "table_eval_doc", "Phase 4/5 table eval doc")

    for doc_id in ranked_docs(doc_stats, key=lambda s: (s["low_table_control_score"], -s["table_caption_count"], -s["table_keyword_block_count"])):
        if bucket_counts["low_table_control"] >= 15 or len(selected) >= args.target_doc_count:
            break
        if doc_stats[doc_id]["chunk_count"] <= 0:
            continue
        add_doc(doc_id, "low_table_control", "low/no-table control document")

    quota_plan = [
        ("high_table_caption", ranked_docs(doc_stats, key=lambda s: (s["table_caption_count"], s["chunk_count"]))),
        ("high_table_keyword", ranked_docs(doc_stats, key=lambda s: (s["table_keyword_block_count"], s["chunk_count"]))),
        ("suspected_table_like_paragraph", ranked_docs(doc_stats, key=lambda s: (s["suspected_table_like_paragraph_count"], s["table_keyword_block_count"]))),
        ("complex_layout", ranked_docs(doc_stats, key=lambda s: (s["complex_score"], s["block_count"]))),
        ("thesis_or_long_doc", ranked_docs(doc_stats, key=lambda s: (s["thesis_long_score"], s["chunk_count"]))),
        ("edge_case_or_miss", sorted(edge_docs)),
        ("low_table_control", ranked_docs(doc_stats, key=lambda s: (s["low_table_control_score"], -s["table_caption_count"]))),
    ]
    bucket_targets = {
        "high_table_caption": 15,
        "high_table_keyword": 15,
        "suspected_table_like_paragraph": 20,
        "complex_layout": 12,
        "thesis_or_long_doc": 10,
        "edge_case_or_miss": 8,
        "low_table_control": 15,
    }
    for bucket, doc_ids in quota_plan:
        for doc_id in doc_ids:
            if len(selected) >= args.target_doc_count:
                break
            if bucket_counts[bucket] >= bucket_targets[bucket]:
                break
            add_doc(doc_id, bucket, reason_for_bucket(bucket))

    if len(selected) < args.target_doc_count:
        for doc_id in ranked_docs(doc_stats, key=lambda s: (s["combined_score"], s["doc_id"])):
            if len(selected) >= args.target_doc_count:
                break
            add_doc(doc_id, "suspected_table_like_paragraph", "deterministic combined representative fill")

    rows = [selected[doc_id] for doc_id in sorted(selected)]
    write_selected(out_dir / "selected_docs.csv", rows)
    write_summary(
        out_dir / "selected_docs_summary.md",
        rows,
        target=args.target_doc_count,
        phase5c1_ids=phase5c1_set,
        phase4_table_docs=phase4_table_docs,
        phase5c2_table_docs=phase5c2_table_docs,
    )
    print(f"selected_docs={len(rows)}")
    print(f"wrote {out_dir / 'selected_docs.csv'}")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def eval_docs(path: Path, query_type: str) -> set[str]:
    return {
        str(item.get("target_doc_id"))
        for item in iter_jsonl(path) or []
        if item.get("query_type") == query_type and item.get("target_doc_id")
    }


def phase4_approved_table_docs(path: Path) -> set[str]:
    docs: set[str] = set()
    for item in iter_jsonl(path) or []:
        sample_type = item.get("sample_type") or item.get("query_type")
        if sample_type == "table" and item.get("include_in_main_denominator", True) is not False:
            if item.get("approved", True) is not False and item.get("target_doc_id"):
                docs.add(str(item["target_doc_id"]))
    return docs


def edge_case_docs(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    docs: set[str] = set()
    for record in payload.get("records", []):
        if record.get("variant") and record.get("variant") != "baseline":
            continue
        if record.get("mode") != "hybrid":
            continue
        if record.get("query_type") in {"table", "table_content", "caption_level_table"} and not record.get("doc_hit_at_10"):
            if record.get("target_doc_id"):
                docs.add(str(record["target_doc_id"]))
    return docs


def accepted_assoc_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in read_csv(path):
        if row.get("accepted_or_rejected") == "accepted" and row.get("doc_id"):
            counts[row["doc_id"]] += 1
    return counts


def baseline_chunk_stats(path: Path) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"chunk_count": 0, "token_count": 0})
    if not path.exists():
        return stats
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            doc_id = str(chunk.get("doc_id") or "")
            if not doc_id:
                continue
            stats[doc_id]["chunk_count"] += 1
            stats[doc_id]["token_count"] += int(chunk.get("token_count") or 0)
    return stats


def collect_doc_stats(
    parsed_clean_dir: Path,
    table_flow: dict[str, dict[str, str]],
    chunks_by_doc: dict[str, dict[str, int]],
    assoc_counts: Counter[str],
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for path in sorted(parsed_clean_dir.glob("doc_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        doc_id = str(data.get("doc_id") or path.stem)
        source_file = str(data.get("source_file") or f"{doc_id}.pdf")
        blocks = [
            block
            for page in data.get("pages", []) or []
            if isinstance(page, dict)
            for block in page.get("blocks", []) or []
            if isinstance(block, dict)
        ]
        type_counts = Counter(str(block.get("type") or "unknown") for block in blocks)
        text_values = [normalize(block.get("text", "")) for block in blocks]
        all_text = "\n".join(text_values[:200])
        table_keyword_count = sum(
            1 for text in text_values if TABLE_KEYWORDS_RE.search(text) or CJK_TABLE_KEYWORDS_RE.search(text)
        )
        flow = table_flow.get(doc_id, {})
        suspected = int_value(flow.get("clean_table_like_paragraph_candidates"))
        if suspected == 0:
            suspected = estimate_table_like_paragraphs(blocks)
        chunk_count = chunks_by_doc.get(doc_id, {}).get("chunk_count", 0)
        total_pages = int_value(data.get("total_pages")) or max([int_value(page.get("page")) for page in data.get("pages", []) if isinstance(page, dict)] or [0])
        cjk = bool(re.search(r"[\u4e00-\u9fff]", all_text))
        block_count = len(blocks)
        table_caption_count = type_counts["table_caption"]
        table_text_count = type_counts["table_text"]
        risk_tags = build_risk_tags(
            cjk=cjk,
            total_pages=total_pages,
            block_count=block_count,
            table_caption_count=table_caption_count,
            table_text_count=table_text_count,
            table_keyword_count=table_keyword_count,
            suspected=suspected,
            assoc_count=assoc_counts[doc_id],
        )
        complex_score = int(block_count >= 160) + int(total_pages >= 14) + int(chunk_count >= 35) + int("dense_page_blocks_complex_layout" in str(flow.get("risk_tags", "")))
        thesis_long_score = int(cjk) * 3 + int(total_pages >= 18) + int(chunk_count >= 40) + int(block_count >= 220)
        low_table_score = int(table_caption_count == 0) * 2 + int(table_keyword_count <= 2) + int(suspected <= 1) + int(chunk_count >= 8)
        combined_score = table_caption_count * 5 + table_keyword_count + suspected * 3 + complex_score * 4 + thesis_long_score * 2
        stats[doc_id] = {
            "doc_id": doc_id,
            "source_file": source_file,
            "table_caption_count": table_caption_count,
            "table_text_count": table_text_count,
            "table_keyword_block_count": table_keyword_count,
            "suspected_table_like_paragraph_count": suspected,
            "chunk_count": chunk_count,
            "risk_tags": ";".join(risk_tags),
            "block_count": block_count,
            "total_pages": total_pages,
            "complex_score": complex_score,
            "thesis_long_score": thesis_long_score,
            "low_table_control_score": low_table_score,
            "combined_score": combined_score,
        }
    return stats


def estimate_table_like_paragraphs(blocks: list[dict[str, Any]]) -> int:
    count = 0
    for block in blocks:
        btype = str(block.get("type") or "")
        if btype not in {"paragraph", "body_text", "list", "list_item", "subsection", "unknown", "text"}:
            continue
        text = normalize(block.get("text", ""))
        if len(text) < 12:
            continue
        numeric = len(re.findall(r"\d+(?:\.\d+)?", text))
        separators = text.count("\t") + text.count("|") + len(re.findall(r"\S\s{2,}\S", text))
        if (numeric >= 3 and (TABLE_KEYWORDS_RE.search(text) or CJK_TABLE_KEYWORDS_RE.search(text))) or separators >= 2:
            count += 1
    return count


def build_risk_tags(
    *,
    cjk: bool,
    total_pages: int,
    block_count: int,
    table_caption_count: int,
    table_text_count: int,
    table_keyword_count: int,
    suspected: int,
    assoc_count: int,
) -> list[str]:
    tags: list[str] = []
    if table_caption_count and not table_text_count:
        tags.append("caption_without_table_text")
    if suspected:
        tags.append("table_like_paragraph_present")
    if assoc_count:
        tags.append("phase5c1_association_present")
    if table_caption_count >= 4:
        tags.append("high_table_caption_count")
    if table_keyword_count >= 20:
        tags.append("high_table_keyword_block_count")
    if block_count >= 180:
        tags.append("dense_blocks_complex_layout")
    if total_pages >= 18:
        tags.append("long_doc")
    if cjk:
        tags.append("cjk_text_present")
    if table_caption_count == 0 and table_keyword_count <= 2:
        tags.append("low_table_control_candidate")
    return tags


def ranked_docs(doc_stats: dict[str, dict[str, Any]], key) -> list[str]:
    return [
        row["doc_id"]
        for row in sorted(doc_stats.values(), key=lambda item: (key(item), item["doc_id"]), reverse=True)
        if positive_score(key(row))
    ]


def positive_score(value: Any) -> bool:
    if isinstance(value, tuple):
        return any(positive_score(item) for item in value)
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return bool(value)


def reason_for_bucket(bucket: str) -> str:
    return {
        "high_table_caption": "high table_caption_count",
        "high_table_keyword": "high table keyword block count",
        "suspected_table_like_paragraph": "many suspected table-like paragraphs",
        "complex_layout": "complex layout/long block distribution proxy",
        "thesis_or_long_doc": "Chinese or thesis-like long document proxy",
        "edge_case_or_miss": "known miss or edge-case retrieval doc",
        "low_table_control": "low/no-table control document",
    }[bucket]


def write_selected(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "doc_id",
        "source_file",
        "reason_selected",
        "from_phase5c1_pilot",
        "table_caption_count",
        "table_text_count",
        "table_keyword_block_count",
        "suspected_table_like_paragraph_count",
        "chunk_count",
        "risk_tags",
        "selection_bucket",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_summary(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    target: int,
    phase5c1_ids: set[str],
    phase4_table_docs: set[str],
    phase5c2_table_docs: set[str],
) -> None:
    buckets = Counter(row["selection_bucket"] for row in rows)
    selected = {row["doc_id"] for row in rows}
    lines = [
        "# Phase 5C-3 Representative Doc Selection",
        "",
        f"- target docs: {target}",
        f"- selected docs: {len(rows)}",
        f"- Phase 5C-1 carryover included: {len(phase5c1_ids & selected)} / {len(phase5c1_ids)}",
        f"- Phase 4 approved table docs included: {len(phase4_table_docs & selected)} / {len(phase4_table_docs)}",
        f"- Phase 5C-2 table_content docs included: {len(phase5c2_table_docs & selected)} / {len(phase5c2_table_docs)}",
        "",
        "## Buckets",
        "",
    ]
    for bucket, count in sorted(buckets.items()):
        lines.append(f"- {bucket}: {count}")
    lines.extend([
        "",
        "## Notes",
        "",
        "- Selection is deterministic and bucketed, not random.",
        "- Full corpus parsed_clean was read only to compute selection metrics; enhancement/chunking/indexing remain scoped to selected docs.",
        "- Low-table controls are included as retrieval takeover checks.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    main()
