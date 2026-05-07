#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.schemas import QueryIntent, RetrievedChunk
from src.synbio_rag.domain.config import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 12E diagnostic smoke for table/figure and comparison slices.")
    parser.add_argument("--table-figure-questions", default="data/evaluation/phase12e_table_figure_diagnostic_questions.jsonl")
    parser.add_argument("--comparison-questions", default="data/evaluation/phase12e_comparison_regression_questions.jsonl")
    parser.add_argument("--output-dir", default="reports/phase12e_diagnostic_smoke")
    parser.add_argument("--parent-index", default="data/paper_round1/chunks/parent_index.jsonl")
    parser.add_argument("--disable-qwen-synthesis", action="store_true", default=True)
    return parser.parse_args()


def load_questions(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_chunk_map(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            rows[str(item.get("chunk_id") or "")] = item
    return rows


def build_settings(args: argparse.Namespace, *, parent_enabled: bool) -> Settings:
    settings = Settings.from_env()
    settings.retrieval.parent_expansion_enabled = parent_enabled
    settings.retrieval.parent_index_path = str((ROOT / args.parent_index).resolve())
    settings.retrieval.milvus_uri = str((ROOT / "runtime/vectorstores/milvus/papers.db").resolve())
    settings.retrieval.collection_name = "synbio_papers"
    settings.kb.chunk_jsonl = str((ROOT / "data/paper_round1/chunks/chunks.jsonl").resolve())
    settings.kb.chunk_dir = str((ROOT / "data/paper_round1/chunks").resolve())
    settings.retrieval.bm25_cache_path = str((ROOT / "data/paper_round1/chunks/bm25_index.json").resolve())
    settings.generation.v2_use_qwen_synthesis = False
    return settings


def answer_preview(text: str, limit: int = 260) -> str:
    cleaned = " ".join((text or "").split())
    return cleaned[:limit]


def normalize_route(route: QueryIntent | str) -> str:
    return getattr(route, "value", str(route))


def contains_chart_terms(question: str) -> bool:
    q = question.lower()
    if any(token in q for token in ["table", "figure", "fig.", "fig ", "chart", "panel"]):
        return True
    if re.search(r"(图|表)\s*\d+", question):
        return True
    return any(token in question for token in ["图表", "表格", "图片", "插图"])


def table_hint_triggered(question: str) -> bool:
    q = question.lower()
    return any(token in q for token in ["table", "primer", "sequence", "strain", "parameter", "表格"])


def figure_hint_triggered(question: str) -> bool:
    q = question.lower()
    return any(token in q for token in ["figure", "fig.", "fig ", "chart", "panel", "图"])


def false_table_trigger_detected(question: str) -> bool:
    q = question.lower()
    if contains_chart_terms(question):
        return False
    return any(token in q for token in ["表达", "expression", "expression cassette", "expression vector", "表征", "表型", "phenotype"])


def run_question(
    pipeline: SynBioRAGPipeline,
    chunk_map: dict[str, dict[str, Any]],
    item: dict[str, Any],
    *,
    parent_enabled: bool,
) -> dict[str, Any]:
    question = item["question"]
    analysis = pipeline.router.analyze(question)
    retrieved, retrieval_debug = pipeline._search_with_filter_fallback(
        question=question,
        analysis=analysis,
        filters=None,
    )
    reranked = pipeline.reranker.rerank(
        question,
        retrieved,
        top_k=analysis.rerank_top_k,
        analysis=analysis,
    )
    seed_chunks = reranked[: pipeline.settings.retrieval.final_top_k]
    final_chunks, parent_debug = pipeline.parent_expander.expand(
        question=question,
        seed_chunks=seed_chunks,
        analysis=analysis,
    )
    gen_result = pipeline.generator_v2.run(
        question=question,
        analysis=analysis,
        seed_chunks=final_chunks,
        config=pipeline.settings.generation,
        history=None,
    )

    seed_ids = [chunk.chunk_id for chunk in seed_chunks]
    final_ids = [chunk.chunk_id for chunk in final_chunks]
    added_ids = [chunk_id for chunk_id in final_ids if chunk_id not in seed_ids]

    added_details: list[dict[str, Any]] = []
    for chunk in final_chunks:
        if chunk.chunk_id not in added_ids:
            continue
        raw = chunk_map.get(chunk.chunk_id, {})
        added_details.append(
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "section": chunk.section,
                "page_numbers": list(raw.get("page_numbers") or (chunk.metadata.get("page_numbers") if isinstance(chunk.metadata, dict) else []) or []),
                "block_types": list(raw.get("block_types") or []),
                "contains_table_caption": bool(raw.get("contains_table_caption")),
                "contains_figure_caption": bool(raw.get("contains_figure_caption")),
                "contains_table_text": bool(raw.get("contains_table_text")),
                "contains_image": bool(raw.get("contains_image")),
                "expanded_from_chunk_id": (chunk.metadata or {}).get("expanded_from_chunk_id"),
                "expanded_from_parent_type": (chunk.metadata or {}).get("expanded_from_parent_type"),
                "parent_expansion_reason": (chunk.metadata or {}).get("parent_expansion_reason"),
            }
        )

    noisy_reason = classify_noisy_reason(
        item=item,
        seed_ids=seed_ids,
        final_ids=final_ids,
        added_details=added_details,
        citations=[citation.chunk_id for citation in (gen_result.citations or [])],
        parent_debug=parent_debug,
        chunk_map=chunk_map,
    )
    judgment = classify_judgment(
        item=item,
        parent_enabled=parent_enabled,
        added_details=added_details,
        noisy_reason=noisy_reason,
    )

    return {
        "id": item["id"],
        "category": item["category"],
        "question": question,
        "expected_doc_id": item.get("expected_doc_id"),
        "anchor_chunk_id": item.get("anchor_chunk_id"),
        "expected_signal": item.get("expected_signal"),
        "expected_behavior": item.get("expected_behavior"),
        "notes": item.get("notes", ""),
        "route": normalize_route(analysis.intent),
        "parent_expansion_enabled": parent_enabled,
        "retrieved_count": len(retrieved),
        "reranked_count": len(reranked),
        "seed_context_count": len(seed_chunks),
        "final_context_count": len(final_chunks),
        "added_count": len(added_ids),
        "parent_types_used": list(parent_debug.get("added_parent_types") or []),
        "seed_chunk_ids": seed_ids,
        "final_chunk_ids": final_ids,
        "added_chunk_ids": added_ids,
        "citation_count": len(gen_result.citations or []),
        "citation_chunk_ids": [citation.chunk_id for citation in (gen_result.citations or [])],
        "answer_preview": answer_preview(gen_result.answer),
        "judgment": judgment,
        "noisy_reason": noisy_reason,
        "added_chunks": added_details,
        "table_hint_triggered": table_hint_triggered(question),
        "figure_hint_triggered": figure_hint_triggered(question),
        "false_table_trigger_detected": false_table_trigger_detected(question),
        "comparison_mode": bool(parent_debug.get("comparison_mode")),
        "effective_max_total": parent_debug.get("effective_max_total"),
        "effective_per_seed_limit": parent_debug.get("effective_per_seed_limit"),
        "comparison_seed_considered": list(parent_debug.get("comparison_seed_considered") or []),
        "comparison_seed_skipped_by_rank": list(parent_debug.get("comparison_seed_skipped_by_rank") or []),
        "per_doc_added": dict(parent_debug.get("per_doc_added") or {}),
        "skipped_by_doc_cap": list(parent_debug.get("skipped_by_doc_cap") or []),
        "selected_parent_types": list(parent_debug.get("selected_parent_types") or []),
        "comparison_caption_allowed": bool(parent_debug.get("comparison_caption_allowed")),
        "caption_mode": bool(parent_debug.get("caption_mode")),
        "figure_query": bool(parent_debug.get("figure_query")),
        "table_query": bool(parent_debug.get("table_query")),
        "caption_query_type": parent_debug.get("caption_query_type"),
        "caption_mode_trigger_source": parent_debug.get("caption_mode_trigger_source"),
        "false_table_trigger_guarded": bool(parent_debug.get("false_table_trigger_guarded")),
        "caption_context_candidates_found": parent_debug.get("caption_context_candidates_found"),
        "caption_context_added": parent_debug.get("caption_context_added"),
        "caption_type_filter": parent_debug.get("caption_type_filter"),
        "caption_candidates_before_type_filter": parent_debug.get("caption_candidates_before_type_filter"),
        "caption_candidates_filtered_by_type": parent_debug.get("caption_candidates_filtered_by_type"),
        "caption_candidates_added_by_type": parent_debug.get("caption_candidates_added_by_type"),
        "caption_seed_docs": list(parent_debug.get("caption_seed_docs") or []),
        "caption_target_doc_ids": list(parent_debug.get("caption_target_doc_ids") or []),
        "skipped_non_target_doc": list(parent_debug.get("skipped_non_target_doc") or []),
        "target_doc_selection_reason": parent_debug.get("target_doc_selection_reason"),
        "page_candidates_found": parent_debug.get("page_candidates_found"),
        "page_candidates_added": parent_debug.get("page_candidates_added"),
        "page_candidates_before_filter": parent_debug.get("page_candidates_before_filter"),
        "page_candidates_filtered_by_doc": parent_debug.get("page_candidates_filtered_by_doc"),
        "page_candidates_filtered_by_type": parent_debug.get("page_candidates_filtered_by_type"),
        "page_plain_paragraph_skipped": parent_debug.get("page_plain_paragraph_skipped"),
        "page_fallback_used": bool(parent_debug.get("page_fallback_used")),
        "same_page_candidates_found": parent_debug.get("same_page_candidates_found"),
        "skipped_cross_doc": parent_debug.get("skipped_cross_doc"),
        "skipped_after_caption_limit": parent_debug.get("skipped_after_caption_limit"),
        "page_skipped_reason": parent_debug.get("page_skipped_reason"),
        "evidence_candidates_found": parent_debug.get("evidence_candidates_found"),
        "evidence_candidates_added": parent_debug.get("evidence_candidates_added"),
        "evidence_skipped_reason": parent_debug.get("evidence_skipped_reason"),
        "parent_expansion_debug": parent_debug,
        "retrieval_debug": retrieval_debug,
    }


def classify_noisy_reason(
    *,
    item: dict[str, Any],
    seed_ids: list[str],
    final_ids: list[str],
    added_details: list[dict[str, Any]],
    citations: list[str],
    parent_debug: dict[str, Any],
    chunk_map: dict[str, dict[str, Any]],
) -> str:
    expected_doc_id = item.get("expected_doc_id")
    expected_signal = item.get("expected_signal")
    if expected_signal == "false_table_trigger":
        if bool(parent_debug.get("caption_mode")) or any(
            parent_type in {"caption_context", "page"} for parent_type in parent_debug.get("added_parent_types") or []
        ):
            return "false_table_trigger"
        return "no_action_needed"

    if expected_doc_id and not any(chunk_id.startswith(expected_doc_id) for chunk_id in seed_ids):
        return "seed_miss"

    if expected_doc_id and any(detail["doc_id"] != expected_doc_id for detail in added_details):
        return "cross_doc_noise"

    if parent_debug.get("skipped_cross_doc", 0) > 0 and expected_doc_id and all(detail["doc_id"] == expected_doc_id for detail in added_details):
        # Skip count is diagnostic only; same-doc guard held.
        pass

    if any(
        detail["expanded_from_parent_type"] == "page"
        and expected_doc_id
        and detail["doc_id"] == expected_doc_id
        and not detail["contains_table_caption"]
        and not detail["contains_figure_caption"]
        and not detail["contains_table_text"]
        for detail in added_details
    ):
        return "page_context_too_broad"

    if any(
        detail["expanded_from_parent_type"] == "caption_context"
        and expected_doc_id
        and detail["doc_id"] == expected_doc_id
        and not detail["contains_table_caption"]
        and not detail["contains_figure_caption"]
        and not detail["contains_table_text"]
        for detail in added_details
    ):
        return "caption_context_noise"

    if added_details and citations:
        citation_docs = {chunk_map.get(chunk_id, {}).get("doc_id", chunk_id.split("_sec")[0]) for chunk_id in citations}
        if expected_doc_id and expected_doc_id not in citation_docs:
            return "generation_not_using_context"

    return "no_action_needed"


def classify_judgment(
    *,
    item: dict[str, Any],
    parent_enabled: bool,
    added_details: list[dict[str, Any]],
    noisy_reason: str,
) -> str:
    if not parent_enabled:
        return "same"
    if noisy_reason in {"cross_doc_noise", "false_table_trigger", "page_context_too_broad", "caption_context_noise"}:
        return "noisy"
    if noisy_reason == "seed_miss":
        return "same"
    if noisy_reason == "generation_not_using_context":
        return "same"
    if added_details:
        if item.get("category") == "comparison":
            return "same"
        return "better"
    return "same"


def run_dataset(
    *,
    questions: list[dict[str, Any]],
    args: argparse.Namespace,
    chunk_map: dict[str, dict[str, Any]],
    prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    a_pipeline = SynBioRAGPipeline(build_settings(args, parent_enabled=False))
    b_pipeline = SynBioRAGPipeline(build_settings(args, parent_enabled=True))
    a_rows = [run_question(a_pipeline, chunk_map, item, parent_enabled=False) for item in questions]
    b_rows = [run_question(b_pipeline, chunk_map, item, parent_enabled=True) for item in questions]
    by_id_a = {row["id"]: row for row in a_rows}
    by_id_b = {row["id"]: row for row in b_rows}
    comparison_rows = []
    for item in questions:
        a = by_id_a[item["id"]]
        b = by_id_b[item["id"]]
        comparison_rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "A": a,
                "B": b,
                "delta_final_context": b["final_context_count"] - a["final_context_count"],
                "delta_citations": b["citation_count"] - a["citation_count"],
            }
        )
    return a_rows, b_rows, comparison_rows


def build_summary(table_b: list[dict[str, Any]], comparison_b: list[dict[str, Any]]) -> dict[str, Any]:
    tf_causes = Counter(row["noisy_reason"] for row in table_b)
    cmp_judgments = Counter(row["judgment"] for row in comparison_b)
    cmp_max_final = max((row["final_context_count"] for row in comparison_b), default=0)
    return {
        "table_figure_count": len(table_b),
        "comparison_count": len(comparison_b),
        "table_figure_judgments": dict(Counter(row["judgment"] for row in table_b)),
        "comparison_judgments": dict(cmp_judgments),
        "table_figure_noisy_causes": dict(tf_causes),
        "comparison_max_final_context": cmp_max_final,
        "comparison_caption_context_hits": sum(1 for row in comparison_b if "caption_context" in row["parent_types_used"]),
        "comparison_page_hits": sum(1 for row in comparison_b if "page" in row["parent_types_used"]),
        "comparison_false_table_triggers": sum(1 for row in comparison_b if row["noisy_reason"] == "false_table_trigger"),
    }


def build_markdown_summary(
    table_b: list[dict[str, Any]],
    comparison_b: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    lines = ["# Phase 12E Diagnostic Smoke", "", "## Table/Figure", ""]
    lines.append(f"- questions: {len(table_b)}")
    lines.append(f"- judgments: {summary['table_figure_judgments']}")
    lines.append(f"- noisy causes: {summary['table_figure_noisy_causes']}")
    lines.append("")
    for row in table_b:
        lines.append(f"- {row['id']}: {row['judgment']} / {row['noisy_reason']} / parents={row['parent_types_used']} / added={row['added_chunk_ids']}")
    lines.extend(["", "## Comparison", ""])
    lines.append(f"- questions: {len(comparison_b)}")
    lines.append(f"- judgments: {summary['comparison_judgments']}")
    lines.append(f"- max_final_context: {summary['comparison_max_final_context']}")
    lines.append("")
    for row in comparison_b:
        lines.append(f"- {row['id']}: {row['judgment']} / final={row['final_context_count']} / per_doc_added={row['per_doc_added']} / parents={row['parent_types_used']}")
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    chunk_map = load_chunk_map(ROOT / "data/paper_round1/chunks/chunks.jsonl")
    table_questions = load_questions(ROOT / args.table_figure_questions)
    comparison_questions = load_questions(ROOT / args.comparison_questions)

    table_a, table_b, table_cmp = run_dataset(
        questions=table_questions,
        args=args,
        chunk_map=chunk_map,
        prefix="table_figure",
    )
    comparison_a, comparison_b, comparison_cmp = run_dataset(
        questions=comparison_questions,
        args=args,
        chunk_map=chunk_map,
        prefix="comparison",
    )
    summary = build_summary(table_b, comparison_b)
    markdown = build_markdown_summary(table_b, comparison_b, summary)

    write_json(output_dir / "phase12e_table_figure_a.json", table_a)
    write_json(output_dir / "phase12e_table_figure_b.json", table_b)
    write_json(output_dir / "phase12e_table_figure_comparison.json", table_cmp)
    write_json(output_dir / "phase12e_comparison_a.json", comparison_a)
    write_json(output_dir / "phase12e_comparison_b.json", comparison_b)
    write_json(output_dir / "phase12e_comparison_comparison.json", comparison_cmp)
    write_json(output_dir / "phase12e_diagnostic_summary.json", summary)
    (output_dir / "phase12e_fix_table_figure_summary.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
