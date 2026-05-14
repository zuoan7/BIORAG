#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit table/figure evidence quality across parsed_clean, evidence units, and chunks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ingestion.document_cleaning_v5 import build_evidence_pack
from src.synbio_rag.ingestion.cleaning_rules import normalize_cleaning_text


EVIDENCE_TYPES = ("table_caption", "table_text", "figure_caption")


def compact_text(text: Any, limit: int = 240) -> str:
    normalized = normalize_cleaning_text(str(text or ""))
    return normalized[:limit]


def iter_json_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("*.json") if path.is_file())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_blocks(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        block
        for page in data.get("pages", []) or []
        if isinstance(page, dict)
        for block in page.get("blocks", []) or []
        if isinstance(block, dict)
    ]


def source_block_id(block: dict[str, Any]) -> str | None:
    metadata = block.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    value = metadata.get("source_block_id") or block.get("source_block_id") or block.get("block_id")
    return str(value) if value is not None else None


def block_id(block: dict[str, Any]) -> str | None:
    value = block.get("block_id") or block.get("id")
    return str(value) if value is not None else None


def page_value(item: dict[str, Any]) -> Any:
    metadata = item.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return item.get("page") or item.get("page_number") or metadata.get("page") or metadata.get("page_number")


def section_path(item: dict[str, Any]) -> list[Any]:
    value = item.get("section_path", []) or []
    return value if isinstance(value, list) else [value]


def load_chunks(chunks_jsonl: Path) -> list[dict[str, Any]]:
    chunks = []
    with chunks_jsonl.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {chunks_jsonl}:{line_num}: {exc}") from exc
            if isinstance(item, dict):
                chunks.append(item)
    return chunks


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if value is None:
        return []
    return [str(value)]


def build_chunk_index(chunks: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        doc_id = str(chunk.get("doc_id", ""))
        keys = set(_as_str_list(chunk.get("block_ids")))
        keys.update(_as_str_list(chunk.get("source_block_ids")))
        for meta in chunk.get("source_block_metadata", []) or []:
            if not isinstance(meta, dict):
                continue
            keys.update(_as_str_list(meta.get("block_id")))
            keys.update(_as_str_list(meta.get("source_block_id")))
        for key in keys:
            if key:
                index[(doc_id, key)].append(chunk)
    return index


def find_chunks_for_item(
    item: dict[str, Any],
    chunk_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    doc_id = str(item.get("doc_id", ""))
    metadata = item.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    keys = set()
    keys.update(_as_str_list(item.get("evidence_id")))
    keys.update(_as_str_list(item.get("block_id")))
    keys.update(_as_str_list(item.get("source_block_id")))
    keys.update(_as_str_list(item.get("block_ids")))
    keys.update(_as_str_list(item.get("source_block_ids")))
    source_meta = metadata.get("source_block_metadata", {}) or {}
    if isinstance(source_meta, dict):
        keys.update(_as_str_list(source_meta.get("block_id")))
        keys.update(_as_str_list(source_meta.get("source_block_id")))

    seen = set()
    matched = []
    for key in keys:
        for chunk in chunk_index.get((doc_id, key), []):
            chunk_id = chunk.get("chunk_id")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            matched.append(chunk)
    return matched


def doc_context(blocks: list[dict[str, Any]], doc_id: str) -> dict[str, Any]:
    by_key: dict[str, dict[str, Any]] = {}
    for index, block in enumerate(blocks):
        enriched = dict(block)
        enriched["_flat_index"] = index
        enriched["doc_id"] = doc_id
        for key in (block_id(block), source_block_id(block)):
            if key:
                by_key[key] = enriched
    return {"blocks": blocks, "by_key": by_key}


def find_source_context(item: dict[str, Any], context: dict[str, Any]) -> tuple[dict[str, Any] | None, str, str]:
    metadata = item.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    source_meta = metadata.get("source_block_metadata", {}) or {}
    if not isinstance(source_meta, dict):
        source_meta = {}

    keys = []
    keys.extend(_as_str_list(item.get("block_id")))
    keys.extend(_as_str_list(item.get("source_block_id")))
    keys.extend(_as_str_list(source_meta.get("block_id")))
    keys.extend(_as_str_list(source_meta.get("source_block_id")))

    source = None
    for key in keys:
        source = context["by_key"].get(key)
        if source:
            break
    if not source:
        return None, "", ""

    blocks = context["blocks"]
    index = source.get("_flat_index")
    previous_preview = ""
    next_preview = ""
    if isinstance(index, int):
        if index > 0:
            previous = blocks[index - 1]
            previous_preview = f"{previous.get('type', '')}: {compact_text(previous.get('text', ''), 180)}"
        if index + 1 < len(blocks):
            next_block = blocks[index + 1]
            next_preview = f"{next_block.get('type', '')}: {compact_text(next_block.get('text', ''), 180)}"
    return source, previous_preview, next_preview


def evidence_pack_for(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("parser_stage") == "evidence_pack_v5" or "evidence_units" in data:
        return data
    return build_evidence_pack(data)


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=lambda item: str(item))}


def bucket_counts(values: list[int]) -> dict[str, int]:
    buckets = Counter(str(value) for value in values)
    return counter_dict(buckets)


def load_preprocess_log(chunks_jsonl: Path, total_docs: int) -> dict[str, Any]:
    failed_log = chunks_jsonl.parent / "failed_docs.log"
    failed_docs = 0
    low_quality_docs = 0
    failed_examples = []
    low_quality_examples = []
    if failed_log.is_file():
        for line in failed_log.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            status = parts[0] if parts else ""
            filename = parts[1] if len(parts) > 1 else ""
            message = parts[2] if len(parts) > 2 else ""
            if status == "FAILED":
                failed_docs += 1
                if len(failed_examples) < 20:
                    failed_examples.append({"source_file": filename, "message": message})
            elif status == "LOW_QUALITY":
                low_quality_docs += 1
                if len(low_quality_examples) < 20:
                    low_quality_examples.append({"source_file": filename, "message": message})

    return {
        "total_docs": total_docs,
        "success_docs": total_docs - failed_docs,
        "failed_docs": failed_docs,
        "low_quality_docs": low_quality_docs,
        "failed_log": str(failed_log),
        "failed_examples": failed_examples,
        "low_quality_examples": low_quality_examples,
    }


def chunk_field_summary(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    if not chunks:
        return {
            "field_names": [],
            "distinct_field_sets": [],
            "distinct_field_set_count": 0,
        }
    field_sets = Counter(tuple(sorted(chunk.keys())) for chunk in chunks)
    return {
        "field_names": list(next(iter(field_sets))),
        "distinct_field_sets": [
            {"count": count, "fields": list(fields)}
            for fields, count in field_sets.most_common()
        ],
        "distinct_field_set_count": len(field_sets),
    }


def _evidence_chunk_flags(chunk: dict[str, Any]) -> tuple[bool, bool, bool]:
    return (
        bool(chunk.get("contains_table_caption")),
        bool(chunk.get("contains_table_text")),
        bool(chunk.get("contains_figure_caption")),
    )


def table_text_alpha_ratio(text: str) -> float:
    stripped = re.sub(r"\[(?:TABLE TEXT|TABLE CAPTION|FIGURE CAPTION)\]", " ", text)
    chars = [char for char in stripped if not char.isspace()]
    if not chars:
        return 0.0
    alpha = sum(1 for char in chars if char.isalpha())
    return alpha / len(chars)


def is_numeric_only_table_chunk(chunk: dict[str, Any]) -> bool:
    text = str(chunk.get("text", ""))
    stripped = re.sub(r"\[(?:TABLE TEXT|TABLE CAPTION|FIGURE CAPTION)\]", " ", text)
    stripped = re.sub(r"[0-9\s.,;:()/%+\-–—≤≥=<>±×*/_]+", "", stripped)
    return not stripped.strip()


def numeric_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "total": 0, "min": None, "max": None, "avg": None}
    return {
        "count": len(values),
        "total": int(sum(values)),
        "min": int(min(values)),
        "max": int(max(values)),
        "avg": sum(values) / len(values),
    }


def chunk_summary(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk.get("chunk_id"),
        "doc_id": chunk.get("doc_id"),
        "source_file": chunk.get("source_file"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "block_types": chunk.get("block_types", []),
        "evidence_types": chunk.get("evidence_types", []),
        "contains_table_caption": bool(chunk.get("contains_table_caption")),
        "contains_table_text": bool(chunk.get("contains_table_text")),
        "contains_figure_caption": bool(chunk.get("contains_figure_caption")),
        "retrieval_text_preview": compact_text(chunk.get("retrieval_text", ""), 280),
        "text_preview": compact_text(chunk.get("text", ""), 280),
    }


def make_sample(
    item: dict[str, Any],
    context: dict[str, Any],
    chunk_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    source, previous_preview, next_preview = find_source_context(item, context)
    chunks = find_chunks_for_item(item, chunk_index)
    first_chunk = chunks[0] if chunks else {}
    metadata = item.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}

    return {
        "doc_id": item.get("doc_id"),
        "source_file": item.get("source_file"),
        "page": page_value(item) if page_value(item) is not None else page_value(source or {}),
        "block_id": item.get("block_id") or (source or {}).get("block_id"),
        "source_block_id": item.get("source_block_id") or source_block_id(source or {}),
        "block_type": (source or item).get("type"),
        "evidence_type": item.get("evidence_type") or item.get("type"),
        "source_clean_block_type": metadata.get("source_clean_block_type"),
        "evidence_type_override": metadata.get("evidence_type_override"),
        "table_group_id": item.get("table_group_id"),
        "figure_group_id": item.get("figure_group_id"),
        "section_path": section_path(item) or section_path(source or {}),
        "text_preview": compact_text(item.get("text", (source or {}).get("text", ""))),
        "chunk_id": first_chunk.get("chunk_id"),
        "chunk_ids": [chunk.get("chunk_id") for chunk in chunks],
        "chunk_retrieval_text_preview": compact_text(first_chunk.get("retrieval_text", ""), 360),
        "previous_block_preview": previous_preview,
        "next_block_preview": next_preview,
    }


def collect_audit(
    parsed_clean_dir: Path,
    chunks_jsonl: Path,
    sample_per_type: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    chunks = load_chunks(chunks_jsonl)
    chunk_index = build_chunk_index(chunks)
    chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[str(chunk.get("doc_id", ""))].append(chunk)

    parsed_counts = Counter()
    evidence_counts = Counter()
    source_clean_counts = Counter()
    table_group_ids = set()
    figure_group_ids = set()
    table_caption_group_ids = set()
    table_text_with_caption = 0
    table_text_without_caption = 0
    table_text_demoted_to_paragraph = 0
    evidence_to_chunk = {
        key: {
            "evidence_units": 0,
            "units_with_chunk": 0,
            "units_without_chunk": 0,
            "chunks": set(),
            "units_per_chunk": Counter(),
        }
        for key in EVIDENCE_TYPES
    }
    evidence_to_chunk["demoted_table_text"] = {
        "evidence_units": 0,
        "units_with_chunk": 0,
        "units_without_chunk": 0,
        "chunks": set(),
        "units_per_chunk": Counter(),
    }

    doc_summaries = []
    samples: dict[str, list[dict[str, Any]]] = {
        "table_caption": [],
        "table_text": [],
        "table_text_without_caption": [],
        "figure_caption": [],
        "demoted_table_text": [],
        "unmatched_evidence": [],
        "mixed_evidence_body_chunks": [],
        "title_unknown_evidence_chunks": [],
        "numeric_low_alpha_table_chunks": [],
    }

    for path in iter_json_files(parsed_clean_dir):
        data = load_json(path)
        doc_id = str(data.get("doc_id", path.stem))
        source_file = str(data.get("source_file", path.name))
        blocks = iter_blocks(data)
        context = doc_context(blocks, doc_id)
        block_counts = Counter(str(block.get("type", "unknown")) for block in blocks)
        parsed_counts.update(block_counts)

        evidence_pack = evidence_pack_for(data)
        evidence_units = [
            dict(unit, doc_id=unit.get("doc_id", doc_id), source_file=unit.get("source_file", source_file))
            for unit in evidence_pack.get("evidence_units", []) or []
            if isinstance(unit, dict)
        ]
        evidence_counts.update(str(unit.get("type", "unknown")) for unit in evidence_units)

        doc_table_groups = set()
        doc_figure_groups = set()
        doc_table_caption_groups = {
            str(unit.get("table_group_id"))
            for unit in evidence_units
            if unit.get("type") == "table_caption" and unit.get("table_group_id")
        }
        table_caption_group_ids.update(doc_table_caption_groups)

        for unit in evidence_units:
            unit_type = str(unit.get("type", "unknown"))
            metadata = unit.get("metadata", {}) or {}
            if not isinstance(metadata, dict):
                metadata = {}
            source_clean_type = metadata.get("source_clean_block_type")
            if source_clean_type:
                source_clean_counts[str(source_clean_type)] += 1

            table_group_id = unit.get("table_group_id")
            figure_group_id = unit.get("figure_group_id")
            if table_group_id:
                table_group_ids.add(str(table_group_id))
                doc_table_groups.add(str(table_group_id))
            if figure_group_id:
                figure_group_ids.add(str(figure_group_id))
                doc_figure_groups.add(str(figure_group_id))

            if source_clean_type == "table_text" and unit_type == "paragraph":
                table_text_demoted_to_paragraph += 1
                sample_key = "demoted_table_text"
            else:
                sample_key = unit_type

            if unit_type == "table_text":
                if table_group_id and str(table_group_id) in doc_table_caption_groups:
                    table_text_with_caption += 1
                else:
                    table_text_without_caption += 1
                    if len(samples["table_text_without_caption"]) < sample_per_type:
                        samples["table_text_without_caption"].append(make_sample(unit, context, chunk_index))

            if sample_key in samples and len(samples[sample_key]) < sample_per_type:
                samples[sample_key].append(make_sample(unit, context, chunk_index))

            distribution_key = sample_key if sample_key == "demoted_table_text" else unit_type
            if distribution_key in evidence_to_chunk:
                matched_chunks = find_chunks_for_item(unit, chunk_index)
                evidence_to_chunk[distribution_key]["evidence_units"] += 1
                if matched_chunks:
                    evidence_to_chunk[distribution_key]["units_with_chunk"] += 1
                else:
                    evidence_to_chunk[distribution_key]["units_without_chunk"] += 1
                    if len(samples["unmatched_evidence"]) < sample_per_type:
                        samples["unmatched_evidence"].append(make_sample(unit, context, chunk_index))
                for chunk in matched_chunks:
                    chunk_id = str(chunk.get("chunk_id", ""))
                    if chunk_id:
                        evidence_to_chunk[distribution_key]["chunks"].add(chunk_id)
                        evidence_to_chunk[distribution_key]["units_per_chunk"][chunk_id] += 1

        doc_summaries.append({
            "path": str(path),
            "doc_id": doc_id,
            "source_file": source_file,
            "parsed_block_counts": counter_dict(block_counts),
            "evidence_type_counts": counter_dict(Counter(str(unit.get("type", "unknown")) for unit in evidence_units)),
            "table_group_id_count": len(doc_table_groups),
            "figure_group_id_count": len(doc_figure_groups),
            "chunk_count": len(chunks_by_doc.get(doc_id, [])),
        })

    chunk_flag_counts = {
        "contains_table_caption": sum(1 for chunk in chunks if chunk.get("contains_table_caption")),
        "contains_table_text": sum(1 for chunk in chunks if chunk.get("contains_table_text")),
        "contains_figure_caption": sum(1 for chunk in chunks if chunk.get("contains_figure_caption")),
        "contains_any_table_or_figure": sum(
            1
            for chunk in chunks
            if chunk.get("contains_table_caption")
            or chunk.get("contains_table_text")
            or chunk.get("contains_figure_caption")
        ),
    }
    chunk_block_type_counts = Counter(
        str(block_type)
        for chunk in chunks
        for block_type in chunk.get("block_types", []) or []
    )
    token_counts = [
        int(chunk.get("token_count", 0))
        for chunk in chunks
        if isinstance(chunk.get("token_count", 0), int)
        or str(chunk.get("token_count", "")).isdigit()
    ]

    mixed_chunks = []
    table_focused_chunks = []
    figure_focused_chunks = []
    caption_with_table_text_chunks = []
    caption_only_table_chunks = []
    orphan_table_text_chunks = []
    evidence_title_section_chunks = []
    evidence_unknown_section_chunks = []
    numeric_only_table_chunks = []
    low_alpha_table_chunks = []
    table_figure_evidence_types = set(EVIDENCE_TYPES)
    for chunk in chunks:
        evidence_types = set(str(item) for item in chunk.get("evidence_types", []) or [])
        block_types = set(str(item) for item in chunk.get("block_types", []) or [])
        has_table_caption, has_table_text, has_figure_caption = _evidence_chunk_flags(chunk)
        has_table_or_figure = bool(evidence_types & table_figure_evidence_types) or any(
            bool(chunk.get(flag))
            for flag in ("contains_table_caption", "contains_table_text", "contains_figure_caption")
        )
        has_body = "paragraph" in block_types or "paragraph" in evidence_types
        has_table = has_table_caption or has_table_text
        if has_table:
            table_focused_chunks.append(chunk)
            alpha_ratio = table_text_alpha_ratio(str(chunk.get("text", "")))
            if is_numeric_only_table_chunk(chunk):
                numeric_only_table_chunks.append(chunk)
            if alpha_ratio < 0.15:
                low_alpha_table_chunks.append(chunk)
            if len(samples["numeric_low_alpha_table_chunks"]) < sample_per_type and (
                alpha_ratio < 0.15 or is_numeric_only_table_chunk(chunk)
            ):
                samples["numeric_low_alpha_table_chunks"].append(chunk_summary(chunk))
        if has_figure_caption:
            figure_focused_chunks.append(chunk)
        if has_table_caption and has_table_text:
            caption_with_table_text_chunks.append(chunk)
        if has_table_caption and not has_table_text:
            caption_only_table_chunks.append(chunk)
        if has_table_text and not has_table_caption:
            orphan_table_text_chunks.append(chunk)
        if has_table_or_figure and str(chunk.get("section", "")) == "Title":
            evidence_title_section_chunks.append(chunk)
            if len(samples["title_unknown_evidence_chunks"]) < sample_per_type:
                samples["title_unknown_evidence_chunks"].append(chunk_summary(chunk))
        if has_table_or_figure and str(chunk.get("section", "")) == "Unknown":
            evidence_unknown_section_chunks.append(chunk)
            if len(samples["title_unknown_evidence_chunks"]) < sample_per_type:
                samples["title_unknown_evidence_chunks"].append(chunk_summary(chunk))
        if has_table_or_figure and has_body:
            mixed_chunks.append(chunk)
            if len(samples["mixed_evidence_body_chunks"]) < sample_per_type:
                samples["mixed_evidence_body_chunks"].append(chunk_summary(chunk))

    distribution = {}
    for key, payload in evidence_to_chunk.items():
        units_per_chunk_values = list(payload["units_per_chunk"].values())
        distribution[key] = {
            "evidence_units": int(payload["evidence_units"]),
            "units_with_chunk": int(payload["units_with_chunk"]),
            "units_without_chunk": int(payload["units_without_chunk"]),
            "chunk_count": len(payload["chunks"]),
            "units_per_chunk_bucket_counts": bucket_counts(units_per_chunk_values),
            "top_chunks_by_unit_count": [
                {"chunk_id": chunk_id, "unit_count": count}
                for chunk_id, count in payload["units_per_chunk"].most_common(20)
            ],
        }

    stats = {
        "inputs": {
            "parsed_clean_dir": str(parsed_clean_dir),
            "chunks_jsonl": str(chunks_jsonl),
            "sample_per_type": sample_per_type,
        },
        "document_count": len(doc_summaries),
        "preprocess": load_preprocess_log(chunks_jsonl, len(doc_summaries)),
        "chunk_count": len(chunks),
        "chunk_schema": chunk_field_summary(chunks),
        "parsed_clean": {
            "block_type_counts": counter_dict(parsed_counts),
            "table_caption_block_count": int(parsed_counts["table_caption"]),
            "figure_caption_block_count": int(parsed_counts["figure_caption"]),
            "table_text_block_count": int(parsed_counts["table_text"]),
        },
        "evidence_units": {
            "evidence_type_counts": counter_dict(evidence_counts),
            "source_clean_block_type_counts": counter_dict(source_clean_counts),
            "table_group_id_count": len(table_group_ids),
            "figure_group_id_count": len(figure_group_ids),
            "table_caption_group_id_count": len(table_caption_group_ids),
            "table_text_with_caption_count": table_text_with_caption,
            "table_text_without_caption_count": table_text_without_caption,
            "table_text_demoted_to_paragraph_count": table_text_demoted_to_paragraph,
        },
        "chunks": {
            **chunk_flag_counts,
            "block_type_counts": counter_dict(chunk_block_type_counts),
            "paragraph_chunk_count": int(chunk_block_type_counts["paragraph"]),
            "token_count_summary": numeric_summary(token_counts),
            "mixed_table_figure_with_paragraph_count": len(mixed_chunks),
            "table_focused_chunk_count": len(table_focused_chunks),
            "figure_focused_chunk_count": len(figure_focused_chunks),
            "caption_with_table_text_chunk_count": len(caption_with_table_text_chunks),
            "caption_only_table_chunk_count": len(caption_only_table_chunks),
            "orphan_table_text_chunk_count": len(orphan_table_text_chunks),
            "evidence_section_title_count": len(evidence_title_section_chunks),
            "evidence_section_unknown_count": len(evidence_unknown_section_chunks),
            "numeric_only_table_chunk_count": len(numeric_only_table_chunks),
            "low_alpha_table_chunk_count": len(low_alpha_table_chunks),
        },
        "evidence_chunk_distribution": distribution,
        "integrity_checks": {
            "has_unmatched_table_figure_evidence": any(
                distribution[key]["units_without_chunk"] > 0 for key in EVIDENCE_TYPES
            ),
            "has_table_figure_evidence_mixed_with_paragraph": len(mixed_chunks) > 0,
            "all_chunks_share_same_field_set": chunk_field_summary(chunks)["distinct_field_set_count"] <= 1,
        },
        "documents": doc_summaries,
    }
    return stats, samples


def render_sample_item(item: dict[str, Any]) -> list[str]:
    lines = [
        f"- doc_id: `{item.get('doc_id', '')}`",
        f"  source_file: `{item.get('source_file', '')}`",
        f"  page: `{item.get('page', '')}`",
        f"  block_id: `{item.get('block_id', '')}`",
        f"  block_type: `{item.get('block_type', '')}`",
        f"  evidence_type: `{item.get('evidence_type', '')}`",
        f"  table_group_id: `{item.get('table_group_id', '')}`",
        f"  figure_group_id: `{item.get('figure_group_id', '')}`",
        f"  section_path: `{json.dumps(item.get('section_path', []), ensure_ascii=False)}`",
        f"  chunk_id: `{item.get('chunk_id', '')}`",
        f"  text preview: {item.get('text_preview', '')}",
    ]
    if item.get("source_clean_block_type") or item.get("evidence_type_override"):
        lines.extend([
            f"  source_clean_block_type: `{item.get('source_clean_block_type', '')}`",
            f"  evidence_type_override: `{item.get('evidence_type_override', '')}`",
        ])
    if item.get("chunk_retrieval_text_preview"):
        lines.append(f"  chunk retrieval preview: {item.get('chunk_retrieval_text_preview', '')}")
    if item.get("previous_block_preview"):
        lines.append(f"  previous: {item.get('previous_block_preview', '')}")
    if item.get("next_block_preview"):
        lines.append(f"  next: {item.get('next_block_preview', '')}")
    lines.append("")
    return lines


def render_chunk_item(item: dict[str, Any]) -> list[str]:
    lines = [
        f"- chunk_id: `{item.get('chunk_id', '')}`",
        f"  doc_id: `{item.get('doc_id', '')}`",
        f"  source_file: `{item.get('source_file', '')}`",
        f"  pages: `{item.get('page_start', '')}-{item.get('page_end', '')}`",
        f"  block_types: `{json.dumps(item.get('block_types', []), ensure_ascii=False)}`",
        f"  evidence_types: `{json.dumps(item.get('evidence_types', []), ensure_ascii=False)}`",
        f"  flags: `table_caption={item.get('contains_table_caption')}, table_text={item.get('contains_table_text')}, figure_caption={item.get('contains_figure_caption')}`",
        f"  text preview: {item.get('text_preview', '')}",
    ]
    if item.get("retrieval_text_preview"):
        lines.append(f"  retrieval preview: {item.get('retrieval_text_preview', '')}")
    lines.append("")
    return lines


def render_markdown(
    stats: dict[str, Any],
    samples: dict[str, list[dict[str, Any]]],
) -> str:
    parsed = stats["parsed_clean"]
    evidence = stats["evidence_units"]
    chunks = stats["chunks"]
    lines = [
        "# Table/Figure Evidence Baseline Audit",
        "",
        f"- parsed_clean_dir: `{stats['inputs']['parsed_clean_dir']}`",
        f"- chunks_jsonl: `{stats['inputs']['chunks_jsonl']}`",
        f"- documents: {stats['document_count']}",
        f"- chunks: {stats['chunk_count']}",
        "",
        "## Summary",
        "",
        f"- success docs: {stats['preprocess']['success_docs']}",
        f"- failed docs: {stats['preprocess']['failed_docs']}",
        f"- low quality docs: {stats['preprocess']['low_quality_docs']}",
        f"- table_caption blocks: {parsed['table_caption_block_count']}",
        f"- figure_caption blocks: {parsed['figure_caption_block_count']}",
        f"- table_text blocks: {parsed['table_text_block_count']}",
        f"- table_group_id count: {evidence['table_group_id_count']}",
        f"- figure_group_id count: {evidence['figure_group_id_count']}",
        f"- table_text with caption: {evidence['table_text_with_caption_count']}",
        f"- table_text without caption: {evidence['table_text_without_caption_count']}",
        f"- table_text demoted to paragraph: {evidence['table_text_demoted_to_paragraph_count']}",
        f"- chunks contains_table_caption: {chunks['contains_table_caption']}",
        f"- chunks contains_table_text: {chunks['contains_table_text']}",
        f"- chunks contains_figure_caption: {chunks['contains_figure_caption']}",
        f"- paragraph chunks: {chunks['paragraph_chunk_count']}",
        f"- total token_count: {chunks['token_count_summary']['total']}",
        f"- mixed table/figure evidence with paragraph chunks: {chunks['mixed_table_figure_with_paragraph_count']}",
        f"- table-focused chunks: {chunks['table_focused_chunk_count']}",
        f"- figure-focused chunks: {chunks['figure_focused_chunk_count']}",
        f"- caption+table_text chunks: {chunks['caption_with_table_text_chunk_count']}",
        f"- caption-only table chunks: {chunks['caption_only_table_chunk_count']}",
        f"- orphan table_text chunks: {chunks['orphan_table_text_chunk_count']}",
        f"- evidence chunks with section Title: {chunks['evidence_section_title_count']}",
        f"- evidence chunks with section Unknown: {chunks['evidence_section_unknown_count']}",
        f"- numeric-only table chunks: {chunks['numeric_only_table_chunk_count']}",
        f"- low-alpha table chunks: {chunks['low_alpha_table_chunk_count']}",
        f"- chunk field set count: {stats['chunk_schema']['distinct_field_set_count']}",
        "",
        "## Evidence Chunk Distribution",
        "",
    ]
    for evidence_type, payload in stats["evidence_chunk_distribution"].items():
        lines.extend([
            f"### {evidence_type}",
            "",
            f"- evidence_units: {payload['evidence_units']}",
            f"- units_with_chunk: {payload['units_with_chunk']}",
            f"- units_without_chunk: {payload['units_without_chunk']}",
            f"- chunk_count: {payload['chunk_count']}",
            f"- units_per_chunk_bucket_counts: `{json.dumps(payload['units_per_chunk_bucket_counts'], ensure_ascii=False)}`",
            "",
        ])

    section_titles = {
        "table_caption": "Table Caption Samples",
        "table_text": "Table Text Samples",
        "table_text_without_caption": "Table Text Without Caption Samples",
        "figure_caption": "Figure Caption Samples",
        "demoted_table_text": "Demoted Table Text Samples",
        "unmatched_evidence": "Unmatched Evidence Samples",
        "mixed_evidence_body_chunks": "Mixed Evidence/Paragraph Chunk Samples",
        "title_unknown_evidence_chunks": "Title/Unknown Evidence Chunk Samples",
        "numeric_low_alpha_table_chunks": "Numeric/Low-Alpha Table Chunk Samples",
    }
    for key, title in section_titles.items():
        lines.extend([f"## {title}", ""])
        if not samples.get(key):
            lines.extend(["No samples.", ""])
            continue
        for item in samples[key]:
            if key in {
                "mixed_evidence_body_chunks",
                "title_unknown_evidence_chunks",
                "numeric_low_alpha_table_chunks",
            }:
                lines.extend(render_chunk_item(item))
            else:
                lines.extend(render_sample_item(item))
    return "\n".join(lines).rstrip() + "\n"


def render_summary(stats: dict[str, Any]) -> str:
    parsed = stats["parsed_clean"]
    preprocess = stats["preprocess"]
    chunks = stats["chunks"]
    distribution = stats["evidence_chunk_distribution"]
    lines = [
        "# Phase 4B Hotfix Full Table/Figure Audit Summary",
        "",
        "## Inputs",
        "",
        f"- parsed_clean_dir: `{stats['inputs']['parsed_clean_dir']}`",
        f"- chunks_jsonl: `{stats['inputs']['chunks_jsonl']}`",
        f"- total_docs: {preprocess['total_docs']}",
        f"- success_docs: {preprocess['success_docs']}",
        f"- failed_docs: {preprocess['failed_docs']}",
        f"- low_quality_docs: {preprocess['low_quality_docs']}",
        f"- chunk_count: {stats['chunk_count']}",
        "",
        "## Evidence Matching",
        "",
    ]
    for key in EVIDENCE_TYPES:
        payload = distribution[key]
        lines.extend([
            f"- {key} blocks: {parsed[f'{key}_block_count']}",
            f"- {key} matched chunks: {payload['chunk_count']}",
            f"- {key} units_with_chunk: {payload['units_with_chunk']}",
            f"- {key} units_without_chunk: {payload['units_without_chunk']}",
        ])
    lines.extend([
        "",
        "## Chunk Separation",
        "",
        f"- mixed_table_figure_with_paragraph_count: {chunks['mixed_table_figure_with_paragraph_count']}",
        f"- table_focused_chunk_count: {chunks['table_focused_chunk_count']}",
        f"- figure_focused_chunk_count: {chunks['figure_focused_chunk_count']}",
        f"- caption_with_table_text_chunk_count: {chunks['caption_with_table_text_chunk_count']}",
        f"- caption_only_table_chunk_count: {chunks['caption_only_table_chunk_count']}",
        f"- orphan_table_text_chunk_count: {chunks['orphan_table_text_chunk_count']}",
        f"- evidence_section_title_count: {chunks['evidence_section_title_count']}",
        f"- evidence_section_unknown_count: {chunks['evidence_section_unknown_count']}",
        f"- numeric_only_table_chunk_count: {chunks['numeric_only_table_chunk_count']}",
        f"- low_alpha_table_chunk_count: {chunks['low_alpha_table_chunk_count']}",
        f"- paragraph_chunk_count: {chunks['paragraph_chunk_count']}",
        f"- token_count_total: {chunks['token_count_summary']['total']}",
        f"- token_count_avg: {chunks['token_count_summary']['avg']}",
        f"- contains_table_caption flag count: {chunks['contains_table_caption']}",
        f"- contains_table_text flag count: {chunks['contains_table_text']}",
        f"- contains_figure_caption flag count: {chunks['contains_figure_caption']}",
        "",
        "## Schema",
        "",
        f"- distinct_field_set_count: {stats['chunk_schema']['distinct_field_set_count']}",
        f"- field_names: `{json.dumps(stats['chunk_schema']['field_names'], ensure_ascii=False)}`",
        "",
        "## Integrity Checks",
        "",
        f"- has_unmatched_table_figure_evidence: {stats['integrity_checks']['has_unmatched_table_figure_evidence']}",
        f"- has_table_figure_evidence_mixed_with_paragraph: {stats['integrity_checks']['has_table_figure_evidence_mixed_with_paragraph']}",
        f"- all_chunks_share_same_field_set: {stats['integrity_checks']['all_chunks_share_same_field_set']}",
    ])
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    output_dir: Path,
    stats: dict[str, Any],
    samples: dict[str, list[dict[str, Any]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "table_figure_stats.json"
    samples_path = output_dir / "table_figure_samples.md"
    summary_path = output_dir / "summary.md"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    samples_path.write_text(render_markdown(stats, samples), encoding="utf-8")
    summary_path.write_text(render_summary(stats), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit table_caption, figure_caption, and table_text evidence in parsed_clean and chunks."
    )
    parser.add_argument("--parsed_clean_dir", required=True, help="Directory containing parsed_clean JSON files.")
    parser.add_argument("--chunks_jsonl", required=True, help="Path to chunks.jsonl.")
    parser.add_argument("--output_dir", required=True, help="Directory for audit outputs.")
    parser.add_argument("--sample_per_type", type=int, default=20, help="Samples per evidence type.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parsed_clean_dir = Path(args.parsed_clean_dir).resolve()
    chunks_jsonl = Path(args.chunks_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not parsed_clean_dir.is_dir():
        raise SystemExit(f"[ERROR] parsed_clean_dir does not exist or is not a directory: {parsed_clean_dir}")
    if not chunks_jsonl.is_file():
        raise SystemExit(f"[ERROR] chunks_jsonl does not exist or is not a file: {chunks_jsonl}")
    if args.sample_per_type < 0:
        raise SystemExit("[ERROR] sample_per_type must be >= 0")

    stats, samples = collect_audit(parsed_clean_dir, chunks_jsonl, args.sample_per_type)
    write_outputs(output_dir, stats, samples)

    print(f"Wrote {output_dir / 'table_figure_stats.json'}")
    print(f"Wrote {output_dir / 'table_figure_samples.md'}")
    print(f"Wrote {output_dir / 'summary.md'}")
    print(
        "Summary: "
        f"table_caption={stats['parsed_clean']['table_caption_block_count']}, "
        f"figure_caption={stats['parsed_clean']['figure_caption_block_count']}, "
        f"table_text={stats['parsed_clean']['table_text_block_count']}, "
        f"chunks_with_table_or_figure={stats['chunks']['contains_any_table_or_figure']}"
    )


if __name__ == "__main__":
    main()
