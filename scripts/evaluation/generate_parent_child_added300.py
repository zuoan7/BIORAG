#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_CHILD_CHUNKS = ROOT / "data/paper_round1/chunks/child_chunks.jsonl"
DEFAULT_PARENT_CHUNKS = ROOT / "data/paper_round1/chunks/parent_chunks.jsonl"
DEFAULT_SMOKE200 = ROOT / "data/eval/datasets/smoke200.jsonl"
DEFAULT_OUTPUT = ROOT / "data/eval/hardsets/parent_child_added300_draft_20260521.jsonl"
DEFAULT_CACHE = ROOT / "results/eval_dataset_generation/parent_child_added300_draft_20260521/cache.jsonl"
DEFAULT_SUMMARY = ROOT / "results/eval_dataset_generation/parent_child_added300_draft_20260521/summary.json"

QUOTAS = {
    "table_content": 80,
    "caption_level_table": 50,
    "figure_caption": 50,
    "normal_factoid": 50,
    "summary_review": 20,
    "comparison": 20,
    "cross_lingual": 20,
    "negative_near_topic": 10,
}

TABLE_HEADING_RE = re.compile(
    r"(^|\n)\s*#{1,4}\s*(Table|TABLE|表)\s*[A-Za-z0-9S.:-]+\b",
    re.M,
)

TOPIC_TERMS = (
    "fucosyllactose",
    "sialyllactose",
    "oligosaccharide",
    "hmo",
    "human milk",
    "pichia",
    "pastoris",
    "yeast",
    "escherichia",
    "e. coli",
    "bacillus",
    "lignin",
    "cellulase",
    "promoter",
    "crispr",
    "pathway",
    "fermentation",
    "biosynthesis",
    "glycosyltransferase",
    "sialyltransferase",
    "fucosyltransferase",
    "protein",
    "enzyme",
)


@dataclass(frozen=True)
class Anchor:
    key: str
    category: str
    route: str
    doc_ids: tuple[str, ...]
    source_files: tuple[str, ...]
    titles: tuple[str, ...]
    sections: tuple[str, ...]
    chunk_ids: tuple[str, ...]
    block_ids: tuple[str, ...]
    texts: tuple[str, ...]
    expected_doc_ids: tuple[str, ...]
    expected_source_files: tuple[str, ...]
    difficulty: str = "hard"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 300 anchored parent-child evaluation questions."
    )
    parser.add_argument("--child-chunks", default=str(DEFAULT_CHILD_CHUNKS))
    parser.add_argument("--parent-chunks", default=str(DEFAULT_PARENT_CHUNKS))
    parser.add_argument("--smoke200", default=str(DEFAULT_SMOKE200))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compact_text(text: str, limit: int = 1400) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 20].rstrip() + " ..."


def is_table_content(row: dict[str, Any]) -> bool:
    text = row.get("text") or ""
    if "[TABLE CAPTION]" in text:
        return False
    if row.get("contains_figure_caption"):
        return False
    return bool(TABLE_HEADING_RE.search(text)) and len(text) > 120


def has_evidence_type(row: dict[str, Any], value: str) -> bool:
    return value in set(row.get("evidence_types") or row.get("block_types") or [])


def clean_body(row: dict[str, Any]) -> bool:
    if has_evidence_type(row, "table_caption") or has_evidence_type(row, "figure_caption"):
        return False
    if row.get("contains_image") or row.get("contains_references") or row.get("contains_metadata"):
        return False
    text = row.get("text") or ""
    if "[TABLE CAPTION]" in text or "[FIGURE CAPTION]" in text:
        return False
    return 100 <= len(text) <= 2500


def smoke_coverage(rows: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    docs: set[str] = set()
    chunks: set[str] = set()
    for row in rows:
        docs.update(str(doc) for doc in row.get("expected_doc_ids") or [])
        meta = row.get("metadata") or {}
        chunks.update(str(chunk) for chunk in meta.get("construction_source_chunk_ids") or [])
        anchor = meta.get("anchor_chunk_id")
        if anchor:
            chunks.add(str(anchor))
    return docs, chunks


def score_row(row: dict[str, Any], smoke_docs: set[str], smoke_chunks: set[str], rng: random.Random) -> tuple[int, int, int, float]:
    doc_score = 0 if row.get("doc_id") not in smoke_docs else 1
    chunk_score = 0 if row.get("chunk_id") not in smoke_chunks else 1
    token_count = int(row.get("token_count") or len((row.get("text") or "").split()))
    length_penalty = abs(token_count - 180)
    return (doc_score, chunk_score, length_penalty, rng.random())


def select_single(
    rows: list[dict[str, Any]],
    *,
    category: str,
    route: str,
    quota: int,
    predicate,
    smoke_docs: set[str],
    smoke_chunks: set[str],
    used_docs: set[str],
    used_chunks: set[str],
    rng: random.Random,
    expected_negative: bool = False,
) -> list[Anchor]:
    candidates = [row for row in rows if predicate(row) and row.get("chunk_id") not in used_chunks]
    candidates.sort(key=lambda row: score_row(row, smoke_docs | used_docs, smoke_chunks, rng))
    selected: list[Anchor] = []

    for row in candidates:
        doc_id = str(row.get("doc_id") or "")
        chunk_id = str(row.get("chunk_id") or "")
        if not doc_id or not chunk_id:
            continue
        if doc_id in used_docs and len(selected) < quota:
            continue
        selected.append(anchor_from_row(row, category, route, expected_negative=expected_negative))
        used_docs.add(doc_id)
        used_chunks.add(chunk_id)
        if len(selected) >= quota:
            break

    if len(selected) < quota:
        for row in candidates:
            chunk_id = str(row.get("chunk_id") or "")
            if chunk_id in used_chunks:
                continue
            selected.append(anchor_from_row(row, category, route, expected_negative=expected_negative))
            used_docs.add(str(row.get("doc_id") or ""))
            used_chunks.add(chunk_id)
            if len(selected) >= quota:
                break

    if len(selected) != quota:
        raise RuntimeError(f"selected {len(selected)} anchors for {category}, expected {quota}")
    return selected


def anchor_from_row(
    row: dict[str, Any],
    category: str,
    route: str,
    *,
    expected_negative: bool = False,
) -> Anchor:
    doc_id = str(row.get("doc_id") or "")
    source_file = str(row.get("source_file") or f"{doc_id}.pdf")
    return Anchor(
        key=f"{category}:{row['chunk_id']}",
        category=category,
        route=route,
        doc_ids=(doc_id,),
        source_files=(source_file,),
        titles=(str(row.get("title") or ""),),
        sections=(str(row.get("section") or ""),),
        chunk_ids=(str(row.get("chunk_id") or ""),),
        block_ids=tuple(str(value) for value in row.get("source_block_ids") or row.get("block_ids") or []),
        texts=(compact_text(str(row.get("text") or "")),),
        expected_doc_ids=() if expected_negative else (doc_id,),
        expected_source_files=() if expected_negative else (source_file,),
        difficulty="hard" if category != "normal_factoid" else "medium",
    )


def topic_set(row: dict[str, Any]) -> set[str]:
    text = " ".join([str(row.get("title") or ""), str(row.get("section") or ""), str(row.get("text") or "")]).lower()
    return {term for term in TOPIC_TERMS if term in text}


def select_comparison(
    parent_rows: list[dict[str, Any]],
    *,
    quota: int,
    smoke_docs: set[str],
    used_docs: set[str],
    used_chunks: set[str],
    rng: random.Random,
) -> list[Anchor]:
    candidates = [
        row
        for row in parent_rows
        if clean_body(row)
        and row.get("doc_id") not in used_docs
        and row.get("doc_id") not in smoke_docs
        and len(topic_set(row)) >= 1
        and 300 <= len(row.get("text") or "") <= 3500
    ]
    candidates.sort(key=lambda row: (len(topic_set(row)) * -1, row.get("doc_id"), rng.random()))

    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        for topic in topic_set(row):
            by_topic[topic].append(row)

    selected: list[Anchor] = []
    for topic, rows in sorted(by_topic.items(), key=lambda item: (-len(item[1]), item[0])):
        rng.shuffle(rows)
        for left in rows:
            if left.get("doc_id") in used_docs or left.get("chunk_id") in used_chunks:
                continue
            right = next(
                (
                    row
                    for row in rows
                    if row.get("doc_id") != left.get("doc_id")
                    and row.get("doc_id") not in used_docs
                    and row.get("chunk_id") not in used_chunks
                ),
                None,
            )
            if right is None:
                continue
            selected.append(
                Anchor(
                    key=f"comparison:{left['chunk_id']}:{right['chunk_id']}",
                    category="comparison",
                    route="comparison",
                    doc_ids=(str(left["doc_id"]), str(right["doc_id"])),
                    source_files=(str(left.get("source_file") or ""), str(right.get("source_file") or "")),
                    titles=(str(left.get("title") or ""), str(right.get("title") or "")),
                    sections=(str(left.get("section") or ""), str(right.get("section") or "")),
                    chunk_ids=(str(left["chunk_id"]), str(right["chunk_id"])),
                    block_ids=tuple(
                        dict.fromkeys(
                            [str(v) for v in left.get("source_block_ids") or []]
                            + [str(v) for v in right.get("source_block_ids") or []]
                        )
                    ),
                    texts=(compact_text(str(left.get("text") or ""), 1100), compact_text(str(right.get("text") or ""), 1100)),
                    expected_doc_ids=(str(left["doc_id"]), str(right["doc_id"])),
                    expected_source_files=(str(left.get("source_file") or ""), str(right.get("source_file") or "")),
                    difficulty="hard",
                )
            )
            used_docs.update([str(left["doc_id"]), str(right["doc_id"])])
            used_chunks.update([str(left["chunk_id"]), str(right["chunk_id"])])
            if len(selected) >= quota:
                return selected

    if len(selected) != quota:
        raise RuntimeError(f"selected {len(selected)} comparison anchors, expected {quota}")
    return selected


def build_anchors(args: argparse.Namespace) -> list[Anchor]:
    rng = random.Random(args.seed)
    child_rows = load_jsonl(args.child_chunks)
    parent_rows = load_jsonl(args.parent_chunks)
    smoke_rows = load_jsonl(args.smoke200)
    smoke_docs, smoke_chunks = smoke_coverage(smoke_rows)

    used_docs: set[str] = set()
    used_chunks: set[str] = set()
    anchors: list[Anchor] = []
    anchors.extend(
        select_single(
            child_rows,
            category="table_content",
            route="factoid",
            quota=QUOTAS["table_content"],
            predicate=is_table_content,
            smoke_docs=smoke_docs,
            smoke_chunks=smoke_chunks,
            used_docs=used_docs,
            used_chunks=used_chunks,
            rng=rng,
        )
    )
    anchors.extend(
        select_single(
            child_rows,
            category="caption_level_table",
            route="factoid",
            quota=QUOTAS["caption_level_table"],
            predicate=lambda row: has_evidence_type(row, "table_caption"),
            smoke_docs=smoke_docs,
            smoke_chunks=smoke_chunks,
            used_docs=used_docs,
            used_chunks=used_chunks,
            rng=rng,
        )
    )
    anchors.extend(
        select_single(
            child_rows,
            category="figure_caption",
            route="factoid",
            quota=QUOTAS["figure_caption"],
            predicate=lambda row: has_evidence_type(row, "figure_caption"),
            smoke_docs=smoke_docs,
            smoke_chunks=smoke_chunks,
            used_docs=used_docs,
            used_chunks=used_chunks,
            rng=rng,
        )
    )
    anchors.extend(
        select_single(
            child_rows,
            category="normal_factoid",
            route="factoid",
            quota=QUOTAS["normal_factoid"],
            predicate=lambda row: clean_body(row) and str(row.get("section") or "") not in {"Title", "Abstract"},
            smoke_docs=smoke_docs,
            smoke_chunks=smoke_chunks,
            used_docs=used_docs,
            used_chunks=used_chunks,
            rng=rng,
        )
    )
    anchors.extend(
        select_single(
            parent_rows,
            category="summary_review",
            route="summary",
            quota=QUOTAS["summary_review"],
            predicate=lambda row: clean_body(row) and 500 <= len(row.get("text") or "") <= 5000,
            smoke_docs=smoke_docs,
            smoke_chunks=smoke_chunks,
            used_docs=used_docs,
            used_chunks=used_chunks,
            rng=rng,
        )
    )
    anchors.extend(
        select_comparison(
            parent_rows,
            quota=QUOTAS["comparison"],
            smoke_docs=smoke_docs,
            used_docs=used_docs,
            used_chunks=used_chunks,
            rng=rng,
        )
    )
    anchors.extend(
        select_single(
            child_rows,
            category="cross_lingual",
            route="factoid",
            quota=QUOTAS["cross_lingual"],
            predicate=lambda row: clean_body(row) and str(row.get("section") or "") in {"Results", "Results and Discussion", "Materials and Methods", "Methods"},
            smoke_docs=smoke_docs,
            smoke_chunks=smoke_chunks,
            used_docs=used_docs,
            used_chunks=used_chunks,
            rng=rng,
        )
    )
    anchors.extend(
        select_single(
            child_rows,
            category="negative_near_topic",
            route="negative",
            quota=QUOTAS["negative_near_topic"],
            predicate=lambda row: clean_body(row) and str(row.get("section") or "") not in {"Title"},
            smoke_docs=smoke_docs,
            smoke_chunks=smoke_chunks,
            used_docs=used_docs,
            used_chunks=used_chunks,
            rng=rng,
            expected_negative=True,
        )
    )
    return anchors


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cache[str(row["key"])] = row
    return cache


def append_cache(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_for_batch(batch: list[Anchor]) -> list[dict[str, str]]:
    payload = []
    for idx, anchor in enumerate(batch, start=1):
        payload.append(
            {
                "id": str(idx),
                "category": anchor.category,
                "expected_route": anchor.route,
                "doc_ids": list(anchor.doc_ids),
                "titles": list(anchor.titles),
                "sections": list(anchor.sections),
                "source_files": list(anchor.source_files),
                "chunk_ids": list(anchor.chunk_ids),
                "evidence": list(anchor.texts),
            }
        )
    user = {
        "task": "Generate one high-quality RAG evaluation question for each anchored evidence item.",
        "hard_rules": [
            "Return JSON only: an array of objects with id, question, rationale.",
            "Do not use a reusable template or merely replace nouns; every question must be written from the specific evidence semantics.",
            "Do not mention chunk IDs, evidence IDs, or that this is an evaluation.",
            "Do not ask for information outside the target document unless category is comparison or negative_near_topic.",
            "Avoid broad questions like 'What does this table show?' unless the evidence is only a caption; prefer concrete entities, variables, methods, or results.",
            "Do not copy a full sentence from evidence; paraphrase naturally.",
            "Make the question answerable by the anchored evidence/doc labels.",
        ],
        "category_guidance": {
            "table_content": "Ask for a specific value, variable relationship, row/column comparison, condition, or measured outcome described in table-like text.",
            "caption_level_table": "Ask a concrete question that should retrieve the table caption or table-level context.",
            "figure_caption": "Ask what a figure reports, compares, validates, or diagrams, using specific entities from the caption.",
            "normal_factoid": "Ask a precise method/result/background question grounded in the evidence.",
            "summary_review": "Ask for a concise synthesis of the study section's method, result pattern, or conclusion.",
            "comparison": "Ask to compare the two target documents on a concrete axis present in both evidence snippets.",
            "cross_lingual": "Write the question in Chinese, while preserving technical terms where useful. The evidence may be English.",
            "negative_near_topic": "Write a plausible near-topic question that asks for a specific detail not supported by the evidence; it should require abstention if no other evidence exists.",
        },
        "items": payload,
    }
    return [
        {
            "role": "system",
            "content": "You create careful biomedical/synthetic-biology RAG evaluation questions. You are precise, non-template, and label-aware.",
        },
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def call_qwen(client: OpenAI, model: str, batch: list[Anchor]) -> list[dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        messages=prompt_for_batch(batch),
        temperature=0.45,
        top_p=0.85,
        max_tokens=2400,
        timeout=90,
    )
    content = response.choices[0].message.content or ""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()
    parsed = json.loads(content)
    if not isinstance(parsed, list):
        raise ValueError("Qwen output is not a JSON array")
    return parsed


def generated_question(client: OpenAI, model: str, anchor: Anchor, cache: dict[str, dict[str, Any]], cache_path: Path) -> dict[str, Any]:
    if anchor.key in cache:
        return cache[anchor.key]
    raise KeyError(anchor.key)


def generate_all(args: argparse.Namespace, anchors: list[Anchor]) -> dict[str, dict[str, Any]]:
    cache_path = Path(args.cache)
    cache = load_cache(cache_path)
    missing = [anchor for anchor in anchors if anchor.key not in cache]
    if args.dry_run:
        print(json.dumps({"anchors": len(anchors), "cached": len(cache), "missing": len(missing), "by_category": Counter(a.category for a in anchors)}, ensure_ascii=False, default=dict))
        return cache

    load_dotenv(ROOT / ".env")
    client = OpenAI(
        api_key=os.getenv("QWEN_CHAT_API_KEY"),
        base_url=os.getenv("QWEN_CHAT_API_BASE"),
    )
    for start in range(0, len(missing), args.batch_size):
        batch = missing[start : start + args.batch_size]
        print(f"generating {start + 1}-{start + len(batch)} / {len(missing)}", flush=True)
        for attempt in range(1, 4):
            try:
                outputs = call_qwen(client, args.model, batch)
                by_id = {str(item.get("id")): item for item in outputs}
                for idx, anchor in enumerate(batch, start=1):
                    item = by_id.get(str(idx))
                    if not item or not str(item.get("question") or "").strip():
                        raise ValueError(f"missing question for batch id {idx}")
                    row = {
                        "key": anchor.key,
                        "question": normalize_question(str(item["question"])),
                        "rationale": str(item.get("rationale") or "").strip(),
                    }
                    append_cache(cache_path, row)
                    cache[anchor.key] = row
                break
            except Exception as exc:
                if attempt == 3:
                    raise
                print(f"retry batch after {type(exc).__name__}: {exc}", flush=True)
                time.sleep(2 * attempt)
    return cache


def normalize_question(question: str) -> str:
    question = re.sub(r"\s+", " ", question).strip()
    question = question.strip("\"'")
    return question


def build_sample(anchor: Anchor, generated: dict[str, Any], index: int, generated_at: str) -> dict[str, Any]:
    sample_id = f"pc_added300_{index:03d}"
    primary_doc = anchor.expected_doc_ids[0] if anchor.expected_doc_ids else ""
    sample = {
        "sample_id": sample_id,
        "query": generated["question"],
        "question": generated["question"],
        "expected_route": anchor.route,
        "expected_doc_ids": list(anchor.expected_doc_ids),
        "expected_source_files": list(anchor.expected_source_files),
        "expected_sections": sorted({section for section in anchor.sections if section}),
        "expected_answer": None,
        "category": anchor.category,
        "query_type": anchor.category,
        "difficulty": anchor.difficulty,
        "source_set": "parent_child_added300_draft",
        "source_dataset": "parent_child_param_ab_added300_20260521",
        "source_sample_id": sample_id,
        "phase_added": "parent_child_param_ab_20260521",
        "tags": tags_for(anchor),
        "expected_behavior": expected_behavior_for(anchor),
        "target_doc_id": primary_doc,
        "target_chunk_id_candidate": anchor.chunk_ids[0] if len(anchor.chunk_ids) == 1 and anchor.expected_doc_ids else "",
        "stable_target_block_ids": list(anchor.block_ids) if anchor.expected_doc_ids else [],
        "metadata": {
            "generation_method": "anchored_semantic_reverse_generation",
            "manual_review_required": True,
            "qwen_model": "qwen-plus",
            "anchor_doc_ids": list(anchor.doc_ids),
            "anchor_source_files": list(anchor.source_files),
            "anchor_titles": list(anchor.titles),
            "anchor_sections": list(anchor.sections),
            "anchor_chunk_ids": list(anchor.chunk_ids),
            "anchor_block_ids": list(anchor.block_ids),
            "construction_source_chunk_ids": list(anchor.chunk_ids),
            "construction_source_doc_ids": list(anchor.doc_ids),
            "label_completeness": "anchored_doc_and_block",
            "rationale": generated.get("rationale", ""),
        },
        "notes": f"Draft anchored {anchor.category} sample generated from latest parent-child chunks; requires spot-check before promotion.",
        "canonical_version": "v1_parent_child_added300_draft",
        "canonicalized_at": generated_at,
        "source_file": "data/paper_round1/chunks/child_chunks.jsonl",
    }
    if anchor.category in {"summary_review", "comparison"}:
        sample["source_file"] = "data/paper_round1/chunks/parent_chunks.jsonl"
    return sample


def tags_for(anchor: Anchor) -> list[str]:
    tags = ["parent_child_added300", "anchored_generation", anchor.category]
    if anchor.category in {"table_content", "caption_level_table", "figure_caption"}:
        tags.append("pdf_table_figure")
    if anchor.category == "cross_lingual":
        tags.extend(["zh_query", "cross_lingual"])
    if anchor.category == "comparison":
        tags.extend(["multi_doc", "comparison"])
    if anchor.category == "negative_near_topic":
        tags.extend(["negative", "near_topic", "abstention"])
    return tags


def expected_behavior_for(anchor: Anchor) -> list[str]:
    if anchor.category == "negative_near_topic":
        return ["abstain_without_direct_evidence", "avoid_wrong_doc_citation"]
    if anchor.category == "comparison":
        return ["balanced_evidence", "requires_citation", "comparison_context_control"]
    return ["requires_citation", "retrieve_anchor_doc"]


def validate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    questions = [sample["question"] for sample in samples]
    duplicate_questions = [q for q, count in Counter(questions).items() if count > 1]
    by_category = Counter(sample["category"] for sample in samples)
    expected = Counter(QUOTAS)
    missing_labels = [
        sample["sample_id"]
        for sample in samples
        if sample["category"] != "negative_near_topic"
        and (not sample["expected_doc_ids"] or not sample["metadata"].get("anchor_chunk_ids"))
    ]
    reused_docs = Counter(
        doc
        for sample in samples
        for doc in sample["metadata"].get("anchor_doc_ids", [])
    )
    return {
        "sample_count": len(samples),
        "by_category": dict(sorted(by_category.items())),
        "expected_by_category": dict(sorted(expected.items())),
        "category_counts_match": by_category == expected,
        "duplicate_question_count": len(duplicate_questions),
        "duplicate_questions": duplicate_questions[:20],
        "missing_label_sample_ids": missing_labels,
        "unique_anchor_doc_count": len(reused_docs),
        "reused_anchor_docs": {doc: count for doc, count in sorted(reused_docs.items()) if count > 1},
    }


def main() -> None:
    args = parse_args()
    anchors = build_anchors(args)
    anchors.sort(key=lambda anchor: list(QUOTAS).index(anchor.category))
    cache = generate_all(args, anchors)
    if args.dry_run:
        return

    generated_at = datetime.now(timezone.utc).isoformat()
    samples = []
    for index, anchor in enumerate(anchors, start=1):
        if anchor.key not in cache:
            raise RuntimeError(f"missing generated question for {anchor.key}")
        samples.append(build_sample(anchor, cache[anchor.key], index, generated_at))

    summary = validate(samples)
    if summary["sample_count"] != sum(QUOTAS.values()) or not summary["category_counts_match"]:
        raise RuntimeError(f"invalid generated set: {summary}")
    if summary["duplicate_question_count"]:
        raise RuntimeError(f"duplicate questions: {summary['duplicate_questions'][:3]}")
    if summary["missing_label_sample_ids"]:
        raise RuntimeError(f"missing labels: {summary['missing_label_sample_ids'][:3]}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    summary.update(
        {
            "output": str(output.resolve()),
            "cache": str(Path(args.cache).resolve()),
            "generated_at": generated_at,
            "content_hash": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    )
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
