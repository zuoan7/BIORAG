#!/usr/bin/env python3
"""Run a Phase7J isolated lexical table retrieval wiring preview."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ELIGIBLE_JSONL = (
    ROOT / "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.jsonl"
)
DEFAULT_QUERY_JSONL = (
    ROOT / "data/experiments/v7_phase7_table_retrieval_wiring_preview/query_set.preview.jsonl"
)
DEFAULT_RESULTS_DIR = ROOT / "results/v7_phase7_table_retrieval_wiring_preview"

TOP_K = 5

RESULT_FIELDS = [
    "query_id",
    "query_text",
    "query_type",
    "rank",
    "matched_table_index_unit_id",
    "matched_unit_type",
    "matched_seed_id",
    "matched_doc_id",
    "matched_table_id",
    "matched_row_label",
    "score",
    "matched_keywords",
    "evidence_text",
    "source_csv_path",
    "source_pdf_crop_path",
    "value_bboxes_available",
    "production_ready",
    "is_expected_seed_hit",
    "is_expected_unit_type_hit",
    "is_expected_unit_hit",
    "match_status",
    "failure_reason",
    "guardrail_limitation",
]

SUMMARY_FIELDS = [
    "query_id",
    "query_type",
    "expected_seed_id",
    "expected_doc_id",
    "expected_table_id",
    "expected_unit_type",
    "expected_table_index_unit_id",
    "expected_row_label",
    "top1_match_status",
    "top1_unit_id",
    "top1_unit_type",
    "top1_score",
    "top1_expected_unit_hit",
    "top3_expected_unit_hit",
    "top5_expected_unit_hit",
    "top1_expected_seed_hit",
    "top3_expected_seed_hit",
    "top5_expected_seed_hit",
    "top1_expected_unit_type_hit",
    "top3_expected_unit_type_hit",
    "top5_expected_unit_type_hit",
    "hit_unit_types",
    "has_no_match",
    "has_weak_match",
]

DEBUG_FIELDS = [
    "query_id",
    "candidate_table_index_unit_id",
    "candidate_unit_type",
    "candidate_seed_id",
    "candidate_doc_id",
    "candidate_table_id",
    "candidate_row_label",
    "rank",
    "score",
    "overlap_count",
    "overlap_tokens",
    "doc_id_phrase_hit",
    "table_id_phrase_hit",
    "row_label_phrase_hit",
    "header_phrase_hits",
    "value_phrase_hits",
    "query_type_unit_boost",
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "find",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "reports",
    "row",
    "table",
    "that",
    "the",
    "this",
    "to",
    "what",
    "where",
    "which",
    "with",
}

QUERY_TYPE_UNIT_BOOST = {
    "table_lookup": "table_unit",
    "row_lookup": "row_unit",
    "metric_lookup": "cell_group_unit",
    "source_or_reference_lookup": "row_unit",
}

GUARDRAIL_LIMITATION = (
    "Phase7J isolated lexical dry-run only; not a formal retrieval evaluation, not a benchmark, "
    "no production index, no external retrieval service."
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalize(value: Any) -> str:
    text = str(value or "").replace("\n", " ")
    return " ".join(text.split())


def norm_for_match(value: Any) -> str:
    return normalize(value).lower()


def tokenize(value: Any) -> list[str]:
    text = norm_for_match(value).replace("−", "-").replace("′", "'")
    raw_tokens = re.findall(r"[a-z0-9]+(?:[_'\-][a-z0-9]+)*|[\u4e00-\u9fff]+", text)
    return [token for token in raw_tokens if len(token) > 1 and token not in STOPWORDS]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def bool_text(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


def value_items(unit: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = unit.get("metadata") or {}
    values = metadata.get("cell_group_values") or metadata.get("row_values") or []
    return [item for item in values if isinstance(item, dict)]


def header_path_text(value: Any) -> str:
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, list):
                parts.append(" / ".join(normalize(part) for part in item if normalize(part)))
            else:
                parts.append(normalize(item))
        return " ".join(part for part in parts if part)
    return normalize(value)


def unit_row_label(unit: dict[str, Any]) -> str:
    return normalize((unit.get("metadata") or {}).get("row_label"))


def unit_search_text(unit: dict[str, Any]) -> str:
    metadata = unit.get("metadata") or {}
    parts: list[str] = [
        unit.get("content_text_for_embedding", ""),
        unit.get("caption", ""),
        unit.get("doc_id", ""),
        unit.get("table_id", ""),
        unit_row_label(unit),
        header_path_text(metadata.get("header_path")),
    ]
    for item in value_items(unit):
        parts.append(item.get("column_header", ""))
        parts.append(header_path_text(item.get("header_path")))
        parts.append(item.get("value", ""))
    return " ".join(normalize(part) for part in parts if normalize(part))


def phrase_hit(query_text: str, phrase: Any) -> bool:
    phrase_norm = norm_for_match(phrase)
    return bool(phrase_norm and phrase_norm in query_text)


def score_unit(query: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    query_text_norm = norm_for_match(query.get("query_text"))
    query_tokens = set(tokenize(query.get("query_text")))
    unit_text = unit_search_text(unit)
    unit_tokens = set(tokenize(unit_text))
    overlap = sorted(query_tokens & unit_tokens)

    score = float(len(overlap) * 2)
    if query_tokens:
        score += 8.0 * (len(overlap) / len(query_tokens))

    doc_hit = phrase_hit(query_text_norm, unit.get("doc_id"))
    table_hit = phrase_hit(query_text_norm, unit.get("table_id"))
    row_hit = phrase_hit(query_text_norm, unit_row_label(unit))
    if doc_hit:
        score += 8.0
    if table_hit:
        score += 6.0
    if row_hit:
        score += 10.0

    header_hits = 0
    value_hits = 0
    for item in value_items(unit):
        header = normalize(item.get("column_header"))
        value = normalize(item.get("value"))
        if len(header) >= 3 and phrase_hit(query_text_norm, header):
            header_hits += 1
        if len(value) >= 3 and phrase_hit(query_text_norm, value):
            value_hits += 1
    if header_hits:
        score += min(header_hits, 4) * 4.0
    if value_hits:
        score += min(value_hits, 3) * 3.0

    query_type_boost = 0.0
    preferred_type = QUERY_TYPE_UNIT_BOOST.get(query.get("query_type"))
    if preferred_type and unit.get("unit_type") == preferred_type:
        query_type_boost = 2.0
        score += query_type_boost

    if not overlap and not any([doc_hit, table_hit, row_hit, header_hits, value_hits]):
        score = 0.0

    return {
        "score": round(score, 6),
        "overlap": overlap,
        "doc_id_phrase_hit": doc_hit,
        "table_id_phrase_hit": table_hit,
        "row_label_phrase_hit": row_hit,
        "header_phrase_hits": header_hits,
        "value_phrase_hits": value_hits,
        "query_type_unit_boost": query_type_boost,
    }


def evidence_text(unit: dict[str, Any]) -> str:
    text = normalize(unit.get("content_text_for_embedding"))
    if text:
        return text
    return normalize(unit.get("content_markdown"))


def match_status(query: dict[str, Any], unit: dict[str, Any] | None, score: float) -> str:
    if unit is None or score <= 0:
        return "no_match"
    if unit.get("table_index_unit_id") == query.get("expected_table_index_unit_id"):
        return "expected_unit_hit"
    if unit.get("seed_id") == query.get("expected_seed_id"):
        return "expected_seed_hit"
    if unit.get("unit_type") == query.get("expected_unit_type"):
        return "expected_type_hit"
    return "weak_match"


def failure_reason_for(status: str) -> str:
    if status == "no_match":
        return "lexical_score_zero"
    if status == "weak_match":
        return "top_k_match_outside_expected_seed_and_type;wiring_sanity_only"
    return ""


def result_row(query: dict[str, Any], unit: dict[str, Any] | None, score_info: dict[str, Any], rank: int) -> dict[str, Any]:
    status = match_status(query, unit, score_info.get("score", 0.0))
    if unit is None:
        return {
            "query_id": query.get("query_id", ""),
            "query_text": query.get("query_text", ""),
            "query_type": query.get("query_type", ""),
            "rank": rank,
            "matched_table_index_unit_id": "",
            "matched_unit_type": "",
            "matched_seed_id": "",
            "matched_doc_id": "",
            "matched_table_id": "",
            "matched_row_label": "",
            "score": "0",
            "matched_keywords": "",
            "evidence_text": "NO_MATCH: lexical token overlap score was zero.",
            "source_csv_path": "",
            "source_pdf_crop_path": "",
            "value_bboxes_available": "",
            "production_ready": "",
            "is_expected_seed_hit": "false",
            "is_expected_unit_type_hit": "false",
            "is_expected_unit_hit": "false",
            "match_status": status,
            "failure_reason": failure_reason_for(status),
            "guardrail_limitation": GUARDRAIL_LIMITATION,
        }

    metadata = unit.get("metadata") or {}
    provenance = unit.get("provenance") or {}
    guardrail = unit.get("guardrail") or {}
    expected_seed_hit = unit.get("seed_id") == query.get("expected_seed_id")
    expected_type_hit = unit.get("unit_type") == query.get("expected_unit_type")
    expected_unit_hit = unit.get("table_index_unit_id") == query.get("expected_table_index_unit_id")
    return {
        "query_id": query.get("query_id", ""),
        "query_text": query.get("query_text", ""),
        "query_type": query.get("query_type", ""),
        "rank": rank,
        "matched_table_index_unit_id": unit.get("table_index_unit_id", ""),
        "matched_unit_type": unit.get("unit_type", ""),
        "matched_seed_id": unit.get("seed_id", ""),
        "matched_doc_id": unit.get("doc_id", ""),
        "matched_table_id": unit.get("table_id", ""),
        "matched_row_label": metadata.get("row_label", "") or "",
        "score": f"{score_info.get('score', 0.0):.6f}",
        "matched_keywords": ";".join(score_info.get("overlap", [])[:20]),
        "evidence_text": evidence_text(unit),
        "source_csv_path": provenance.get("source_csv_path", ""),
        "source_pdf_crop_path": provenance.get("source_pdf_crop_path", ""),
        "value_bboxes_available": bool_text(provenance.get("value_bboxes_available")),
        "production_ready": bool_text(guardrail.get("production_ready")),
        "is_expected_seed_hit": bool_text(expected_seed_hit),
        "is_expected_unit_type_hit": bool_text(expected_type_hit),
        "is_expected_unit_hit": bool_text(expected_unit_hit),
        "match_status": status,
        "failure_reason": failure_reason_for(status),
        "guardrail_limitation": GUARDRAIL_LIMITATION,
    }


def no_match_score_info() -> dict[str, Any]:
    return {
        "score": 0.0,
        "overlap": [],
        "doc_id_phrase_hit": False,
        "table_id_phrase_hit": False,
        "row_label_phrase_hit": False,
        "header_phrase_hits": 0,
        "value_phrase_hits": 0,
        "query_type_unit_boost": 0.0,
    }


def rank_units(query: dict[str, Any], units: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    scored = [(unit, score_unit(query, unit)) for unit in units]
    scored.sort(
        key=lambda item: (
            item[1]["score"],
            item[0].get("seed_id") == query.get("expected_seed_id"),
            item[0].get("unit_type") == query.get("expected_unit_type"),
            item[0].get("table_index_unit_id") == query.get("expected_table_index_unit_id"),
            item[0].get("table_index_unit_id", ""),
        ),
        reverse=True,
    )
    return scored


def build_query_summary(query: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked_rows = [row for row in rows if row["query_id"] == query["query_id"]]
    top1 = ranked_rows[0] if ranked_rows else {}

    def any_hit(key: str, k: int) -> bool:
        return any(row.get(key) == "true" for row in ranked_rows[:k])

    hit_unit_types = [
        row["matched_unit_type"]
        for row in ranked_rows
        if row.get("matched_unit_type") and row.get("match_status") != "no_match"
    ]
    return {
        "query_id": query["query_id"],
        "query_type": query["query_type"],
        "expected_seed_id": query.get("expected_seed_id", ""),
        "expected_doc_id": query.get("expected_doc_id", ""),
        "expected_table_id": query.get("expected_table_id", ""),
        "expected_unit_type": query.get("expected_unit_type", ""),
        "expected_table_index_unit_id": query.get("expected_table_index_unit_id", ""),
        "expected_row_label": query.get("expected_row_label", ""),
        "top1_match_status": top1.get("match_status", ""),
        "top1_unit_id": top1.get("matched_table_index_unit_id", ""),
        "top1_unit_type": top1.get("matched_unit_type", ""),
        "top1_score": top1.get("score", ""),
        "top1_expected_unit_hit": bool_text(any_hit("is_expected_unit_hit", 1)),
        "top3_expected_unit_hit": bool_text(any_hit("is_expected_unit_hit", 3)),
        "top5_expected_unit_hit": bool_text(any_hit("is_expected_unit_hit", 5)),
        "top1_expected_seed_hit": bool_text(any_hit("is_expected_seed_hit", 1)),
        "top3_expected_seed_hit": bool_text(any_hit("is_expected_seed_hit", 3)),
        "top5_expected_seed_hit": bool_text(any_hit("is_expected_seed_hit", 5)),
        "top1_expected_unit_type_hit": bool_text(any_hit("is_expected_unit_type_hit", 1)),
        "top3_expected_unit_type_hit": bool_text(any_hit("is_expected_unit_type_hit", 3)),
        "top5_expected_unit_type_hit": bool_text(any_hit("is_expected_unit_type_hit", 5)),
        "hit_unit_types": ";".join(sorted(set(hit_unit_types))),
        "has_no_match": bool_text(any(row.get("match_status") == "no_match" for row in ranked_rows)),
        "has_weak_match": bool_text(any(row.get("match_status") == "weak_match" for row in ranked_rows)),
    }


def debug_row(
    query: dict[str, Any],
    unit: dict[str, Any],
    score_info: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    return {
        "query_id": query.get("query_id", ""),
        "candidate_table_index_unit_id": unit.get("table_index_unit_id", ""),
        "candidate_unit_type": unit.get("unit_type", ""),
        "candidate_seed_id": unit.get("seed_id", ""),
        "candidate_doc_id": unit.get("doc_id", ""),
        "candidate_table_id": unit.get("table_id", ""),
        "candidate_row_label": unit_row_label(unit),
        "rank": rank,
        "score": f"{score_info.get('score', 0.0):.6f}",
        "overlap_count": len(score_info.get("overlap", [])),
        "overlap_tokens": ";".join(score_info.get("overlap", [])[:30]),
        "doc_id_phrase_hit": bool_text(score_info.get("doc_id_phrase_hit")),
        "table_id_phrase_hit": bool_text(score_info.get("table_id_phrase_hit")),
        "row_label_phrase_hit": bool_text(score_info.get("row_label_phrase_hit")),
        "header_phrase_hits": score_info.get("header_phrase_hits", 0),
        "value_phrase_hits": score_info.get("value_phrase_hits", 0),
        "query_type_unit_boost": score_info.get("query_type_unit_boost", 0.0),
    }


def run_preview(
    eligible_jsonl: Path = DEFAULT_ELIGIBLE_JSONL,
    query_jsonl: Path = DEFAULT_QUERY_JSONL,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    eligible_jsonl = resolve_path(eligible_jsonl)
    query_jsonl = resolve_path(query_jsonl)
    results_dir = resolve_path(results_dir)

    units = load_jsonl(eligible_jsonl)
    queries = load_jsonl(query_jsonl)
    eligible_ids = {unit["table_index_unit_id"] for unit in units}
    if len(units) != 274 and eligible_jsonl == DEFAULT_ELIGIBLE_JSONL:
        raise ValueError(f"expected 274 eligible units, got {len(units)}")

    result_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    topk_records: list[dict[str, Any]] = []

    for query in queries:
        ranked = rank_units(query, units)
        positive = [(unit, info) for unit, info in ranked if info["score"] > 0]
        query_rows: list[dict[str, Any]] = []
        if not positive:
            row = result_row(query, None, no_match_score_info(), 1)
            query_rows.append(row)
            result_rows.append(row)
        else:
            for rank, (unit, score_info) in enumerate(positive[:top_k], start=1):
                if unit["table_index_unit_id"] not in eligible_ids:
                    raise ValueError("matched unit outside eligible set")
                row = result_row(query, unit, score_info, rank)
                query_rows.append(row)
                result_rows.append(row)
            for rank, (unit, score_info) in enumerate(positive[: max(top_k, 10)], start=1):
                debug_rows.append(debug_row(query, unit, score_info, rank))

        topk_records.append(
            {
                "query_id": query.get("query_id", ""),
                "query_text": query.get("query_text", ""),
                "query_type": query.get("query_type", ""),
                "expected_seed_id": query.get("expected_seed_id", ""),
                "expected_doc_id": query.get("expected_doc_id", ""),
                "expected_table_id": query.get("expected_table_id", ""),
                "expected_unit_type": query.get("expected_unit_type", ""),
                "expected_table_index_unit_id": query.get("expected_table_index_unit_id", ""),
                "expected_row_label": query.get("expected_row_label", ""),
                "top_k": query_rows,
                "guardrail_limitation": GUARDRAIL_LIMITATION,
            }
        )

    query_summaries = [build_query_summary(query, result_rows) for query in queries]
    match_status_counts = Counter(row["match_status"] for row in result_rows)
    hit_unit_type_counts = Counter(
        row["matched_unit_type"]
        for row in result_rows
        if row.get("matched_unit_type") and row.get("match_status") != "no_match"
    )
    query_type_counts = Counter(query["query_type"] for query in queries)

    results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(results_dir / "retrieval_wiring_preview_results.csv", result_rows, RESULT_FIELDS)
    write_jsonl(results_dir / "topk_evidence_units.jsonl", topk_records)
    write_csv(results_dir / "query_hit_summary.csv", query_summaries, SUMMARY_FIELDS)
    write_csv(results_dir / "ranking_debug.csv", debug_rows, DEBUG_FIELDS)

    top1_expected_unit_hits = sum(
        1 for row in query_summaries if row["top1_expected_unit_hit"] == "true"
    )
    top3_expected_unit_hits = sum(
        1 for row in query_summaries if row["top3_expected_unit_hit"] == "true"
    )
    top5_expected_unit_hits = sum(
        1 for row in query_summaries if row["top5_expected_unit_hit"] == "true"
    )
    summary = {
        "phase": "v7_phase7J",
        "preview_name": "offline_table_retrieval_wiring_preview",
        "official_benchmark": False,
        "formal_retrieval_evaluation": False,
        "top_k": top_k,
        "eligible_unit_count": len(units),
        "query_count": len(queries),
        "result_row_count": len(result_rows),
        "query_type_distribution": dict(query_type_counts),
        "match_status_counts": dict(match_status_counts),
        "hit_unit_type_counts": dict(hit_unit_type_counts),
        "top1_expected_unit_hit_queries": top1_expected_unit_hits,
        "top3_expected_unit_hit_queries": top3_expected_unit_hits,
        "top5_expected_unit_hit_queries": top5_expected_unit_hits,
        "no_match_count": match_status_counts.get("no_match", 0),
        "weak_match_count": match_status_counts.get("weak_match", 0),
        "guardrail": {
            "only_phase7i1_eligible_units": True,
            "excluded_units_used_for_matching": False,
            "isolated_lexical_matching_only": True,
            "bm25_index_read_or_queried": False,
            "milvus_accessed_or_written": False,
            "embedding_run": False,
            "rerank_run": False,
            "model_called": False,
            "production_index_built": False,
            "flat_chunks_compared": False,
            "official_benchmark_written": False,
            "production_connected": False,
        },
        "generated_files": [
            rel(results_dir / "retrieval_wiring_preview_results.csv"),
            rel(results_dir / "retrieval_wiring_preview_results.json"),
            rel(results_dir / "topk_evidence_units.jsonl"),
            rel(results_dir / "query_hit_summary.csv"),
            rel(results_dir / "ranking_debug.csv"),
        ],
        "query_summaries": query_summaries,
        "records": result_rows,
    }
    (results_dir / "retrieval_wiring_preview_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if any(math.isnan(float(row["score"])) for row in result_rows if row["score"]):
        raise ValueError("nan score detected")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligible-jsonl", type=Path, default=DEFAULT_ELIGIBLE_JSONL)
    parser.add_argument("--query-jsonl", type=Path, default=DEFAULT_QUERY_JSONL)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_preview(
        eligible_jsonl=args.eligible_jsonl,
        query_jsonl=args.query_jsonl,
        results_dir=args.results_dir,
        top_k=args.top_k,
    )
    printable = {key: value for key, value in summary.items() if key not in {"records", "query_summaries"}}
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
