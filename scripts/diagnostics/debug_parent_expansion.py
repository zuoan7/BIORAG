from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.synbio_rag.application.parent_expansion import ParentContextExpander
from src.synbio_rag.domain.config import RetrievalConfig
from src.synbio_rag.domain.schemas import QueryAnalysis, QueryIntent, RetrievedChunk
from src.synbio_rag.infrastructure.index.parent_store import ParentStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run parent expansion without Milvus or LLM.")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--parents", required=True)
    parser.add_argument("--seed-chunk-ids", nargs="+", required=True)
    parser.add_argument("--intent", choices=["factoid", "summary", "comparison"], default="factoid")
    parser.add_argument("--question", required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    store = ParentStore.from_jsonl(args.parents, chunk_jsonl_path=args.chunks)
    config = RetrievalConfig(
        parent_expansion_enabled=True,
        parent_index_path=args.parents,
    )
    expander = ParentContextExpander(store, config)

    seeds: list[RetrievedChunk] = []
    for chunk_id in args.seed_chunk_ids:
        chunk = store.get_chunk(chunk_id)
        if chunk is None:
            continue
        seeds.append(chunk)

    analysis = QueryAnalysis(
        intent=QueryIntent(args.intent),
        requires_external_tools=False,
        search_limit=0,
        rerank_top_k=0,
        notes="debug_parent_expansion",
    )
    final_chunks, debug = expander.expand(args.question, seeds, analysis)
    added = [chunk.chunk_id for chunk in final_chunks if chunk.chunk_id not in {seed.chunk_id for seed in seeds}]
    print(json.dumps({
        "seed_chunk_ids": [chunk.chunk_id for chunk in seeds],
        "final_chunk_ids": [chunk.chunk_id for chunk in final_chunks],
        "added_chunk_ids": added,
        "added_parent_types": debug.get("added_parent_types", []),
        "parent_expansion": debug,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
