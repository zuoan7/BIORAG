from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.synbio_rag.infrastructure.index.parent_store import ParentStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check parent index sidecar contract.")
    parser.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--parents", required=True, help="Path to parent_index.jsonl")
    return parser


def load_chunks(path: str) -> tuple[list[dict], set[str]]:
    rows: list[dict] = []
    chunk_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            rows.append(item)
            chunk_id = str(item.get("chunk_id") or "")
            if chunk_id:
                chunk_ids.add(chunk_id)
    return rows, chunk_ids


def load_parents(path: str) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            records.append(json.loads(raw))
    return records


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    chunks, chunk_ids = load_chunks(args.chunks)
    parents = load_parents(args.parents)
    store = ParentStore.from_jsonl(args.parents, chunk_jsonl_path=args.chunks)

    chunk_by_id = {str(chunk.get("chunk_id") or ""): chunk for chunk in chunks if str(chunk.get("chunk_id") or "").strip()}
    parent_type_counts = Counter(str(parent.get("parent_type") or "") for parent in parents)
    evidence_type_counts = Counter(
        str(parent.get("evidence_type") or "")
        for parent in parents
        if parent.get("parent_type") == "evidence_type_context"
    )

    orphan_child_ids: set[str] = set()
    membership: dict[str, set[str]] = defaultdict(set)
    empty_parent_count = 0
    caption_same_page_children = 0
    caption_non_anchor_children = 0
    caption_child_lengths = Counter()

    for parent in parents:
        parent_type = str(parent.get("parent_type") or "")
        child_chunk_ids = [str(v) for v in parent.get("child_chunk_ids") or [] if str(v or "").strip()]
        if not child_chunk_ids:
            empty_parent_count += 1
        for chunk_id in child_chunk_ids:
            membership[chunk_id].add(parent_type)
            if chunk_id not in chunk_ids:
                orphan_child_ids.add(chunk_id)

        if parent_type == "caption_context":
            caption_child_lengths[len(child_chunk_ids)] += 1
            anchor_chunk_id = str(parent.get("anchor_chunk_id") or "")
            anchor_pages = set((chunk_by_id.get(anchor_chunk_id) or {}).get("page_numbers") or [])
            for child_chunk_id in child_chunk_ids:
                if child_chunk_id == anchor_chunk_id:
                    continue
                child_pages = set((chunk_by_id.get(child_chunk_id) or {}).get("page_numbers") or [])
                caption_non_anchor_children += 1
                if anchor_pages and child_pages & anchor_pages:
                    caption_same_page_children += 1

    missing_core_membership = sum(
        1
        for chunk_id in chunk_ids
        if not {"doc", "section", "section_path", "chunk_window"} <= membership.get(chunk_id, set())
    )

    same_page_ratio = (
        caption_same_page_children / caption_non_anchor_children
        if caption_non_anchor_children
        else 0.0
    )

    print(f"total_chunks: {len(chunks)}")
    print(f"total_parents: {len(parents)}")
    print(f"parent_type_distribution: {dict(parent_type_counts)}")
    print(f"orphan_child_count: {len(orphan_child_ids)}")
    print(f"parents_with_zero_children: {empty_parent_count}")
    print(f"chunks_missing_doc_section_section_path_chunk_window_membership: {missing_core_membership}")
    print(f"doc_parent_count: {parent_type_counts.get('doc', 0)}")
    print(f"section_parent_count: {parent_type_counts.get('section', 0)}")
    print(f"section_path_parent_count: {parent_type_counts.get('section_path', 0)}")
    print(f"page_parent_count: {parent_type_counts.get('page', 0)}")
    print(f"chunk_window_parent_count: {parent_type_counts.get('chunk_window', 0)}")
    print(f"caption_context_parent_count: {parent_type_counts.get('caption_context', 0)}")
    print(f"evidence_type_context_parent_count: {parent_type_counts.get('evidence_type_context', 0)}")
    print(f"evidence_type_distribution: {dict(evidence_type_counts)}")
    print(f"caption_context_child_length_distribution: {dict(sorted(caption_child_lengths.items()))}")
    print(f"caption_context_same_page_child_ratio: {same_page_ratio:.4f}")
    print(f"parent_store_loadable: {bool(store)}")

    rng = random.Random(0)
    _print_samples("section_path_samples", [p for p in parents if p.get("parent_type") == "section_path"], rng, 3)
    _print_samples("page_samples", [p for p in parents if p.get("parent_type") == "page"], rng, 3)
    _print_samples("caption_context_samples", [p for p in parents if p.get("parent_type") == "caption_context"], rng, 3)
    _print_samples(
        "evidence_type_context_samples",
        [p for p in parents if p.get("parent_type") == "evidence_type_context"],
        rng,
        3,
    )

    return 0


def _print_samples(label: str, records: list[dict], rng: random.Random, count: int) -> None:
    print(f"{label}:")
    sample_size = min(count, len(records))
    for record in rng.sample(records, sample_size) if sample_size else []:
        print(
            json.dumps(
                {
                    "parent_id": record.get("parent_id"),
                    "doc_id": record.get("doc_id"),
                    "parent_type": record.get("parent_type"),
                    "section": record.get("section"),
                    "section_path": record.get("section_path"),
                    "section_path_key": record.get("section_path_key"),
                    "page_number": record.get("page_number"),
                    "anchor_chunk_id": record.get("anchor_chunk_id"),
                    "caption_kind": record.get("caption_kind"),
                    "evidence_type": record.get("evidence_type"),
                    "child_chunk_ids": record.get("child_chunk_ids"),
                    "page_numbers": record.get("page_numbers"),
                    "text_preview": record.get("text_preview"),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
