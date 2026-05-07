#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.config import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small A/B smoke for parent expansion.")
    parser.add_argument(
        "--questions",
        default="data/evaluation/phase12d_small_smoke_questions.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/phase12d_small_smoke",
    )
    parser.add_argument(
        "--disable-qwen-synthesis",
        action="store_true",
        help="Turn off v2 qwen synthesis to reduce nondeterminism in A/B smoke.",
    )
    parser.add_argument(
        "--parent-index",
        default="data/paper_round1/chunks/parent_index.jsonl",
    )
    parser.add_argument(
        "--output-prefix",
        default="phase12d_small_smoke",
    )
    return parser.parse_args()


def load_questions(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def answer_preview(text: str, limit: int = 220) -> str:
    cleaned = " ".join((text or "").split())
    return cleaned[:limit]


def build_settings(args: argparse.Namespace, *, parent_enabled: bool) -> Settings:
    settings = Settings.from_env()
    settings.retrieval.parent_expansion_enabled = parent_enabled
    settings.retrieval.parent_index_path = str(Path(args.parent_index).resolve())
    settings.retrieval.milvus_uri = str((ROOT / "runtime/vectorstores/milvus/papers.db").resolve())
    settings.retrieval.collection_name = "synbio_papers"
    settings.kb.chunk_jsonl = str((ROOT / "data/paper_round1/chunks/chunks.jsonl").resolve())
    settings.kb.chunk_dir = str((ROOT / "data/paper_round1/chunks").resolve())
    settings.retrieval.bm25_cache_path = str((ROOT / "data/paper_round1/chunks/bm25_index.json").resolve())
    if args.disable_qwen_synthesis:
        settings.generation.v2_use_qwen_synthesis = False
    return settings


def normalize_route(route: Any) -> str:
    return getattr(route, "value", str(route))


def run_mode(
    questions: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    parent_enabled: bool,
) -> list[dict[str, Any]]:
    settings = build_settings(args, parent_enabled=parent_enabled)
    pipeline = SynBioRAGPipeline(settings)
    rows: list[dict[str, Any]] = []
    for item in questions:
        response = pipeline.answer(item["question"])
        dbg = response.debug or {}
        px = dbg.get("parent_expansion", {}) or {}
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "route": normalize_route(response.route),
                "parent_expansion_enabled": parent_enabled,
                "retrieved_count": dbg.get("retrieved_count"),
                "reranked_count": dbg.get("reranked_count"),
                "seed_context_count": dbg.get("seed_context_count"),
                "final_context_count": dbg.get("final_context_count"),
                "citations_count": len(response.citations or []),
                "answer_preview": answer_preview(response.answer),
                "citation_chunk_ids": [c.chunk_id for c in (response.citations or [])],
                "parent_expansion": {
                    "enabled": px.get("enabled"),
                    "reason": px.get("reason"),
                    "input_count": px.get("input_count"),
                    "output_count": px.get("output_count"),
                    "added_count": len(px.get("added_chunk_ids", []) or []),
                    "parent_types_used": px.get("added_parent_types", []),
                    "added_chunk_ids": px.get("added_chunk_ids", []),
                    "per_doc_added": px.get("per_doc_added", {}),
                    "effective_intent": px.get("effective_intent"),
                    "effective_max_total": px.get("effective_max_total"),
                    "effective_per_seed_limit": px.get("effective_per_seed_limit"),
                    "limit_reason": px.get("limit_reason"),
                    "comparison_mode": px.get("comparison_mode"),
                    "comparison_seed_considered": px.get("comparison_seed_considered", []),
                    "comparison_seed_skipped_by_rank": px.get("comparison_seed_skipped_by_rank", []),
                    "skipped_by_doc_cap": px.get("skipped_by_doc_cap", []),
                    "comparison_caption_allowed": px.get("comparison_caption_allowed"),
                    "caption_mode": px.get("caption_mode"),
                    "caption_anchor_doc_id": px.get("caption_anchor_doc_id"),
                    "same_doc_only": px.get("same_doc_only"),
                    "same_page_candidates_found": px.get("same_page_candidates_found"),
                    "caption_context_candidates_found": px.get("caption_context_candidates_found"),
                    "caption_context_added": px.get("caption_context_added"),
                    "page_context_added": px.get("page_context_added"),
                    "skipped_cross_doc": px.get("skipped_cross_doc"),
                    "skipped_after_caption_limit": px.get("skipped_after_caption_limit"),
                    "page_candidates_found": px.get("page_candidates_found"),
                    "page_candidates_added": px.get("page_candidates_added"),
                    "page_skipped_reason": px.get("page_skipped_reason"),
                    "evidence_candidates_found": px.get("evidence_candidates_found"),
                    "evidence_candidates_added": px.get("evidence_candidates_added"),
                    "evidence_skipped_reason": px.get("evidence_skipped_reason"),
                    "summary_docs_considered": px.get("summary_docs_considered", []),
                    "summary_sections_added": px.get("summary_sections_added", []),
                    "summary_sections_skipped_existing": px.get("summary_sections_skipped_existing", []),
                    "summary_no_candidate_docs": px.get("summary_no_candidate_docs", []),
                },
            }
        )
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_comparison(
    questions: list[dict[str, Any]],
    a_rows: list[dict[str, Any]],
    b_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id_a = {row["id"]: row for row in a_rows}
    by_id_b = {row["id"]: row for row in b_rows}
    combined: list[dict[str, Any]] = []
    for item in questions:
        a_row = by_id_a[item["id"]]
        b_row = by_id_b[item["id"]]
        combined.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "A": a_row,
                "B": b_row,
                "delta_final_context": (b_row["final_context_count"] or 0) - (a_row["final_context_count"] or 0),
                "delta_citations": (b_row["citations_count"] or 0) - (a_row["citations_count"] or 0),
            }
        )
    return combined


def main() -> None:
    args = parse_args()
    questions_path = Path(args.questions)
    output_dir = Path(args.output_dir)
    questions = load_questions(questions_path)
    a_rows = run_mode(questions, args, parent_enabled=False)
    b_rows = run_mode(questions, args, parent_enabled=True)
    comparison = build_comparison(questions, a_rows, b_rows)
    summary = {
        "questions": len(questions),
        "qwen_synthesis_disabled": bool(args.disable_qwen_synthesis),
        "control_parent_expansion_enabled": False,
        "treatment_parent_expansion_enabled": True,
        "with_added_chunks": sum(1 for row in b_rows if row["parent_expansion"]["added_count"] > 0),
        "caption_parent_hits": sum(
            1 for row in b_rows if "caption_context" in row["parent_expansion"]["parent_types_used"]
        ),
        "page_parent_hits": sum(
            1 for row in b_rows if "page" in row["parent_expansion"]["parent_types_used"]
        ),
        "section_path_hits": sum(
            1 for row in b_rows if "section_path" in row["parent_expansion"]["parent_types_used"]
        ),
        "evidence_parent_hits": sum(
            1 for row in b_rows if "evidence_type_context" in row["parent_expansion"]["parent_types_used"]
        ),
    }
    write_json(output_dir / f"{args.output_prefix}_a.json", a_rows)
    write_json(output_dir / f"{args.output_prefix}_b.json", b_rows)
    write_json(output_dir / f"{args.output_prefix}_comparison.json", comparison)
    write_json(output_dir / f"{args.output_prefix}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
