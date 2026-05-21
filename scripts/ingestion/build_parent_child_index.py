from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ingestion.build_parent_index import build_parent_records, make_parent_id


DEFAULT_CHILD_SIZE = 160
DEFAULT_CHILD_OVERLAP = 40
MAX_HEADING_WORDS = 80

HEADING_TYPES = {"title", "section_heading", "subsection_heading"}
TABLE_TYPES = {"table_caption", "table_text"}
FIGURE_TYPES = {"figure_caption", "image_caption"}
EVIDENCE_TYPES = TABLE_TYPES | FIGURE_TYPES


@dataclass(frozen=True)
class BlockSegment:
    text: str
    block_type: str
    block_id: str = ""
    source_block_id: str = ""
    page: int | None = None
    section_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    block_index: int = 0
    token_start: int = 0
    token_end: int = 0
    split_from_long_block: bool = False


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for ordinal, raw in enumerate(handle):
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            item.setdefault("_ordinal", ordinal)
            rows.append(item)
    return rows


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_parent_child_records(
    chunks: list[dict[str, Any]],
    *,
    child_size: int = DEFAULT_CHILD_SIZE,
    child_overlap: int = DEFAULT_CHILD_OVERLAP,
    window_size: int = 1,
    caption_context_max_children: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if child_size <= 0:
        raise ValueError("child_size must be positive")
    if child_overlap < 0:
        raise ValueError("child_overlap must be non-negative")
    if child_overlap >= child_size:
        raise ValueError("child_overlap must be smaller than child_size")

    parent_chunks: list[dict[str, Any]] = []
    child_chunks: list[dict[str, Any]] = []
    retrieval_parent_records: list[dict[str, Any]] = []

    for chunk in sorted(chunks, key=_chunk_sort_key):
        parent_chunk = dict(chunk)
        parent_chunk_id = str(parent_chunk.get("chunk_id") or "")
        if not parent_chunk_id:
            continue
        doc_id = str(parent_chunk.get("doc_id") or "")
        parent_id = make_parent_id(
            "retrieval_parent",
            doc_id,
            f"{doc_id}::retrieval_parent::{parent_chunk_id}",
        )
        parent_chunk["index_role"] = "parent"
        parent_chunk["parent_id"] = parent_id
        parent_chunk["parent_chunk_id"] = parent_chunk_id
        parent_chunks.append(parent_chunk)

        children = _split_parent_chunk(
            parent_chunk=parent_chunk,
            parent_id=parent_id,
            child_size=child_size,
            child_overlap=child_overlap,
        )
        child_chunks.extend(children)
        retrieval_parent_records.append(
            _build_retrieval_parent_record(
                parent_chunk=parent_chunk,
                parent_id=parent_id,
                child_chunk_ids=[child["chunk_id"] for child in children],
                child_size=child_size,
                child_overlap=child_overlap,
            )
        )

    context_parent_records = build_parent_records(
        parent_chunks,
        window_size=window_size,
        caption_context_max_children=caption_context_max_children,
    )
    return parent_chunks, child_chunks, context_parent_records + retrieval_parent_records


def _split_parent_chunk(
    *,
    parent_chunk: dict[str, Any],
    parent_id: str,
    child_size: int,
    child_overlap: int,
) -> list[dict[str, Any]]:
    parent_chunk_id = str(parent_chunk.get("chunk_id") or "")
    segments = _segments_from_parent_chunk(parent_chunk)
    if not segments:
        return []

    child_groups = _build_structural_child_groups(
        segments=segments,
        child_size=child_size,
        child_overlap=child_overlap,
    )
    children: list[dict[str, Any]] = []
    for child_index, group in enumerate(child_groups, start=1):
        child_text = "\n\n".join(segment.text for segment in group if segment.text.strip())
        if not child_text.strip():
            continue
        child = dict(parent_chunk)
        child["chunk_id"] = f"{parent_chunk_id}::child{child_index:03d}"
        child["text"] = child_text
        child["retrieval_text"] = _build_child_retrieval_text(parent_chunk, child_text)
        child["token_count"] = len(child_text.split())
        child["chunk_index"] = _child_sort_index(parent_chunk, child_index)
        _apply_child_structural_metadata(child, group, parent_chunk)
        child["index_role"] = "child"
        child["parent_id"] = parent_id
        child["parent_chunk_id"] = parent_chunk_id
        child["child_index"] = child_index
        child["child_start_token"] = min(segment.token_start for segment in group)
        child["child_end_token"] = max(segment.token_end for segment in group)
        child["child_token_count"] = len(child_text.split())
        child["child_split_strategy"] = _child_split_strategy(group)
        children.append(child)
    return children


def _segments_from_parent_chunk(parent_chunk: dict[str, Any]) -> list[BlockSegment]:
    text = str(parent_chunk.get("text") or "")
    parts = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not parts:
        return []

    metadata_items = [
        item for item in parent_chunk.get("source_block_metadata") or []
        if isinstance(item, dict)
    ]
    matched_metadata = _align_metadata_to_text_parts(parts, metadata_items)
    segments: list[BlockSegment] = []
    token_cursor = 0
    for index, part in enumerate(parts):
        meta = matched_metadata[index] if index < len(matched_metadata) else {}
        block_type = _segment_block_type(part, meta)
        token_count = len(part.split())
        segment = BlockSegment(
            text=part,
            block_type=block_type,
            block_id=str(meta.get("block_id") or ""),
            source_block_id=str(meta.get("source_block_id") or meta.get("block_id") or ""),
            page=_safe_int(meta.get("page")),
            section_path=_coerce_str_list(meta.get("section_path")),
            metadata=meta,
            block_index=index,
            token_start=token_cursor,
            token_end=token_cursor + token_count,
        )
        segments.append(segment)
        token_cursor += token_count
    return segments


def _align_metadata_to_text_parts(
    parts: list[str],
    metadata_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not metadata_items:
        return [{} for _part in parts]

    aligned: list[dict[str, Any]] = []
    cursor = 0
    for part_index, part in enumerate(parts):
        part_norm = _normalize_for_alignment(part)
        selected: dict[str, Any] = {}
        for meta_index in range(cursor, len(metadata_items)):
            preview = str(metadata_items[meta_index].get("text_preview") or "")
            preview_norm = _normalize_for_alignment(preview)
            if _metadata_preview_matches(part_norm, preview_norm):
                selected = metadata_items[meta_index]
                cursor = meta_index + 1
                break
        if (
            not selected
            and len(parts) == len(metadata_items)
            and cursor < len(metadata_items)
        ):
            selected = metadata_items[cursor]
            cursor += 1
        if (
            not selected
            and cursor < len(metadata_items)
            and not _metadata_matches_later_part(metadata_items[cursor], parts[part_index + 1 :])
        ):
            cursor += 1
        aligned.append(selected)
    return aligned


def _metadata_matches_later_part(metadata: dict[str, Any], parts: list[str]) -> bool:
    preview_norm = _normalize_for_alignment(metadata.get("text_preview") or "")
    if not preview_norm:
        return False
    return any(
        _metadata_preview_matches(_normalize_for_alignment(part), preview_norm)
        for part in parts
    )


def _metadata_preview_matches(part_norm: str, preview_norm: str) -> bool:
    if not part_norm or not preview_norm:
        return False
    part_head = part_norm[:80]
    preview_head = preview_norm[:80]
    return part_head.startswith(preview_head[:40]) or preview_head.startswith(part_head[:40])


def _normalize_for_alignment(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _segment_block_type(text: str, metadata: dict[str, Any]) -> str:
    block_type = str(metadata.get("type") or "").strip()
    if block_type:
        if block_type in HEADING_TYPES and len(str(text or "").split()) > MAX_HEADING_WORDS:
            return "paragraph"
        return block_type
    stripped = text.strip()
    lowered = stripped.lower()
    if stripped.startswith("###"):
        return "subsection_heading"
    if stripped.startswith("##") or lowered in {"abstract", "introduction", "results", "discussion", "conclusion"}:
        return "section_heading"
    if stripped.startswith("#"):
        return "title"
    if re.match(r"^(?:table|tbl\\.)\\s*\\d+", stripped, re.I):
        return "table_caption"
    if re.match(r"^(?:fig\\.?|figure)\\s*\\d+", stripped, re.I):
        return "figure_caption"
    return "paragraph"


def _build_structural_child_groups(
    *,
    segments: list[BlockSegment],
    child_size: int,
    child_overlap: int,
) -> list[list[BlockSegment]]:
    groups: list[list[BlockSegment]] = []
    pending_headings: list[BlockSegment] = []
    body_group: list[BlockSegment] = []

    def flush_body() -> None:
        nonlocal body_group
        if body_group:
            groups.append(body_group)
            body_group = []

    index = 0
    while index < len(segments):
        segment = segments[index]
        if segment.block_type in HEADING_TYPES:
            pending_headings.append(segment)
            index += 1
            continue

        if segment.block_type in EVIDENCE_TYPES:
            flush_body()
            evidence_group = list(pending_headings)
            pending_headings = []
            evidence_group.append(segment)
            if segment.block_type in TABLE_TYPES:
                index += 1
                while index < len(segments) and segments[index].block_type in TABLE_TYPES:
                    evidence_group.append(segments[index])
                    index += 1
                groups.extend(_split_large_structural_group(evidence_group, child_size, child_overlap))
                continue
            groups.extend(_split_large_structural_group(evidence_group, child_size, child_overlap))
            index += 1
            continue

        unit = list(pending_headings)
        pending_headings = []
        unit.append(segment)
        unit_tokens = _group_token_count(unit)
        if unit_tokens > child_size and len(unit) == 1:
            flush_body()
            groups.extend(_split_long_segment(segment, child_size, child_overlap))
            index += 1
            continue
        if unit_tokens > child_size and len(unit) > 1:
            flush_body()
            groups.extend(_split_large_structural_group(unit, child_size, child_overlap))
            index += 1
            continue
        if body_group and _group_token_count(body_group) + unit_tokens > child_size:
            flush_body()
        body_group.extend(unit)
        index += 1

    flush_body()
    if pending_headings:
        groups.extend(_split_large_structural_group(pending_headings, child_size, child_overlap))
    return groups


def _split_large_structural_group(
    group: list[BlockSegment],
    child_size: int,
    child_overlap: int,
) -> list[list[BlockSegment]]:
    group_tokens = _group_token_count(group)
    if group_tokens <= child_size:
        return [group]
    if any(segment.block_type in EVIDENCE_TYPES for segment in group) and group_tokens <= child_size * 2:
        return [group]
    if len(group) == 1:
        return _split_long_segment(group[0], child_size, child_overlap)

    headings = [segment for segment in group if segment.block_type in HEADING_TYPES]
    body_segments = [segment for segment in group if segment.block_type not in HEADING_TYPES]
    if not body_segments:
        return _pack_structural_segments(group, child_size, child_overlap)
    if _group_token_count(headings) > MAX_HEADING_WORDS:
        return _pack_structural_segments(group, child_size, child_overlap)

    output: list[list[BlockSegment]] = []
    current: list[BlockSegment] = list(headings)
    for segment in body_segments:
        segment_tokens = len(segment.text.split())
        if segment_tokens > child_size:
            if current and current != headings:
                output.append(current)
                current = list(headings)
            split_segments = _split_long_segment(segment, child_size, child_overlap)
            for split_group in split_segments:
                output.append(list(headings) + split_group if headings else split_group)
            continue
        if current and current != headings and _group_token_count(current) + segment_tokens > child_size:
            output.append(current)
            current = list(headings)
        current.append(segment)
    if current and (current != headings or not output):
        output.append(current)
    return output


def _pack_structural_segments(
    segments: list[BlockSegment],
    child_size: int,
    child_overlap: int,
) -> list[list[BlockSegment]]:
    output: list[list[BlockSegment]] = []
    current: list[BlockSegment] = []
    for segment in segments:
        segment_tokens = len(segment.text.split())
        if segment_tokens > child_size:
            if current:
                output.append(current)
                current = []
            output.extend(_split_long_segment(segment, child_size, child_overlap))
            continue
        if current and _group_token_count(current) + segment_tokens > child_size:
            output.append(current)
            current = []
        current.append(segment)
    if current:
        output.append(current)
    return output


def _split_long_segment(
    segment: BlockSegment,
    child_size: int,
    child_overlap: int,
) -> list[list[BlockSegment]]:
    words = segment.text.split()
    if not words:
        return []
    spans = _word_spans(len(words), child_size, child_overlap)
    split_groups: list[list[BlockSegment]] = []
    for split_index, (start, end) in enumerate(spans, start=1):
        split_text = " ".join(words[start:end])
        split_segment = BlockSegment(
            text=split_text,
            block_type=segment.block_type,
            block_id=segment.block_id,
            source_block_id=segment.source_block_id,
            page=segment.page,
            section_path=list(segment.section_path),
            metadata=dict(segment.metadata),
            block_index=segment.block_index,
            token_start=segment.token_start + start,
            token_end=segment.token_start + end,
            split_from_long_block=True,
        )
        split_groups.append([split_segment])
    return split_groups


def _group_token_count(group: list[BlockSegment]) -> int:
    return sum(len(segment.text.split()) for segment in group)


def _apply_child_structural_metadata(
    child: dict[str, Any],
    group: list[BlockSegment],
    parent_chunk: dict[str, Any],
) -> None:
    block_types = _ordered_unique(segment.block_type for segment in group if segment.block_type)
    block_ids = _ordered_unique(segment.block_id for segment in group if segment.block_id)
    source_block_ids = _ordered_unique(segment.source_block_id for segment in group if segment.source_block_id)
    page_numbers = _ordered_unique_ints(segment.page for segment in group if segment.page is not None)
    section_path = _first_non_empty_section_path(group) or _coerce_str_list(parent_chunk.get("section_path"))

    child["block_types"] = block_types
    child["block_ids"] = block_ids
    child["source_block_ids"] = source_block_ids
    child["page_numbers"] = page_numbers
    child["page_start"] = min(page_numbers) if page_numbers else parent_chunk.get("page_start")
    child["page_end"] = max(page_numbers) if page_numbers else parent_chunk.get("page_end")
    child["section_path"] = section_path
    child["source_block_metadata"] = [_child_source_metadata(segment) for segment in group]
    child["evidence_types"] = _child_evidence_types(block_types, parent_chunk)
    child["contains_table_caption"] = "table_caption" in block_types
    child["contains_table_text"] = "table_text" in block_types
    child["contains_figure_caption"] = bool(set(block_types) & FIGURE_TYPES)
    child["contains_image"] = "image" in block_types or "image_caption" in block_types
    child["contains_references"] = "references" in block_types
    child["contains_metadata"] = "metadata" in block_types
    child["contains_noise"] = "noise" in block_types
    child["child_block_start_index"] = min(segment.block_index for segment in group)
    child["child_block_end_index"] = max(segment.block_index for segment in group)


def _child_source_metadata(segment: BlockSegment) -> dict[str, Any]:
    metadata = dict(segment.metadata)
    metadata.setdefault("block_id", segment.block_id)
    metadata.setdefault("source_block_id", segment.source_block_id)
    metadata["type"] = segment.block_type
    if segment.page is not None:
        metadata.setdefault("page", segment.page)
    if segment.section_path:
        metadata.setdefault("section_path", list(segment.section_path))
    metadata["child_text_preview"] = _compact_preview(segment.text, limit=160)
    return metadata


def _first_non_empty_section_path(group: list[BlockSegment]) -> list[str]:
    for segment in group:
        if segment.section_path:
            return list(segment.section_path)
    return []


def _child_evidence_types(block_types: list[str], parent_chunk: dict[str, Any]) -> list[str]:
    structural = [block_type for block_type in block_types if block_type not in {"overlap"}]
    if structural:
        return _ordered_unique(structural)
    return _coerce_str_list(parent_chunk.get("evidence_types"))


def _child_split_strategy(group: list[BlockSegment]) -> str:
    if any(segment.split_from_long_block for segment in group):
        return "long_block_window"
    if any(segment.block_type in EVIDENCE_TYPES for segment in group):
        return "evidence_block"
    return "structure_block"


def _word_spans(total_words: int, child_size: int, child_overlap: int) -> list[tuple[int, int]]:
    if total_words <= child_size:
        return [(0, total_words)]
    spans: list[tuple[int, int]] = []
    step = max(1, child_size - child_overlap)
    start = 0
    while start < total_words:
        end = min(total_words, start + child_size)
        spans.append((start, end))
        if end >= total_words:
            break
        start += step
    return spans


def _build_child_retrieval_text(parent_chunk: dict[str, Any], child_text: str) -> str:
    header_parts: list[str] = []
    if parent_chunk.get("title"):
        header_parts.append(f"title: {parent_chunk['title']}")
    if parent_chunk.get("section"):
        header_parts.append(f"section: {parent_chunk['section']}")
    if parent_chunk.get("source_file"):
        header_parts.append(f"source_file: {parent_chunk['source_file']}")
    if parent_chunk.get("doc_id"):
        header_parts.append(f"doc_id: {parent_chunk['doc_id']}")
    evidence_types = parent_chunk.get("evidence_types") or []
    if evidence_types:
        header_parts.append(f"evidence_type: {', '.join(sorted(set(evidence_types)))}")
    if not header_parts:
        return child_text
    return "\n".join(header_parts) + "\n\n" + child_text


def _build_retrieval_parent_record(
    *,
    parent_chunk: dict[str, Any],
    parent_id: str,
    child_chunk_ids: list[str],
    child_size: int,
    child_overlap: int,
) -> dict[str, Any]:
    parent_chunk_id = str(parent_chunk.get("chunk_id") or "")
    page_numbers = _coerce_int_list(parent_chunk.get("page_numbers"))
    return {
        "parent_id": parent_id,
        "parent_type": "retrieval_parent",
        "doc_id": str(parent_chunk.get("doc_id") or ""),
        "parent_chunk_id": parent_chunk_id,
        "source_file": str(parent_chunk.get("source_file") or ""),
        "title": str(parent_chunk.get("title") or ""),
        "section": str(parent_chunk.get("section") or ""),
        "section_path": _coerce_str_list(parent_chunk.get("section_path")),
        "section_path_key": _normalize_section_path(_coerce_str_list(parent_chunk.get("section_path"))),
        "anchor_chunk_id": parent_chunk_id,
        "child_chunk_ids": child_chunk_ids,
        "page_start": _safe_int(parent_chunk.get("page_start")),
        "page_end": _safe_int(parent_chunk.get("page_end")),
        "page_numbers": page_numbers,
        "page_number": None,
        "block_ids": _coerce_str_list(parent_chunk.get("block_ids")),
        "source_block_ids": _coerce_str_list(parent_chunk.get("source_block_ids")),
        "content_kinds": _derive_content_kinds(parent_chunk),
        "contains_table_caption": bool(parent_chunk.get("contains_table_caption")),
        "contains_figure_caption": bool(parent_chunk.get("contains_figure_caption")),
        "contains_table_text": bool(parent_chunk.get("contains_table_text")),
        "contains_image": bool(parent_chunk.get("contains_image")),
        "text_preview": _compact_preview(parent_chunk.get("text")),
        "caption_kind": _caption_kind(parent_chunk),
        "evidence_type": "",
        "child_chunk_size": child_size,
        "child_chunk_overlap": child_overlap,
    }


def _derive_content_kinds(chunk: dict[str, Any]) -> list[str]:
    kinds: list[str] = []
    if chunk.get("contains_table_text"):
        kinds.append("table_text")
    if chunk.get("contains_table_caption"):
        kinds.append("table_caption")
    if chunk.get("contains_figure_caption"):
        kinds.append("figure_caption")
    if chunk.get("contains_image"):
        kinds.append("image_related")
    if chunk.get("contains_references"):
        kinds.append("references")
    if chunk.get("contains_metadata"):
        kinds.append("metadata")
    return kinds or ["body"]


def _caption_kind(chunk: dict[str, Any]) -> str:
    if chunk.get("contains_table_caption") and chunk.get("contains_figure_caption"):
        return "mixed"
    if chunk.get("contains_table_caption"):
        return "table_caption"
    if chunk.get("contains_figure_caption"):
        return "figure_caption"
    return ""


def _child_sort_index(parent_chunk: dict[str, Any], child_index: int) -> int:
    parent_index = _safe_int(parent_chunk.get("chunk_index"), 0) or 0
    return parent_index * 1000 + child_index


def _chunk_sort_key(item: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(item.get("doc_id") or ""),
        _safe_int(item.get("chunk_index"), item.get("_ordinal", 0)) or 0,
        str(item.get("chunk_id") or ""),
    )


def _compact_preview(value: object, limit: int = 280) -> str:
    compact = " ".join(str(value or "").split())
    return compact[:limit]


def _normalize_section_path(section_path: list[str]) -> str:
    return " > ".join(part.strip() for part in section_path if part and part.strip())


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if str(v or "").strip()]


def _coerce_int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    output: list[int] = []
    for item in value:
        coerced = _safe_int(item)
        if coerced is not None:
            output.append(coerced)
    return output


def _ordered_unique(values: Any) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not value:
            continue
        value_str = str(value)
        if value_str in seen:
            continue
        seen.add(value_str)
        output.append(value_str)
    return output


def _ordered_unique_ints(values: Any) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for value in values:
        coerced = _safe_int(value)
        if coerced is None or coerced in seen:
            continue
        seen.add(coerced)
        output.append(coerced)
    return output


def _safe_int(value: object, default: int | None = None) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build true parent-child retrieval artifacts from parent chunks.")
    parser.add_argument("--chunks", required=True, help="Source parent chunks JSONL.")
    parser.add_argument("--parent-output", required=True, help="Output parent_chunks.jsonl.")
    parser.add_argument("--child-output", required=True, help="Output child_chunks.jsonl for dense/BM25 retrieval.")
    parser.add_argument("--parent-index-output", required=True, help="Output parent_index.jsonl.")
    parser.add_argument("--child-size", type=int, default=DEFAULT_CHILD_SIZE)
    parser.add_argument("--child-overlap", type=int, default=DEFAULT_CHILD_OVERLAP)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--caption-context-max-children", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    chunks = load_jsonl(args.chunks)
    parent_chunks, child_chunks, parent_index = build_parent_child_records(
        chunks,
        child_size=args.child_size,
        child_overlap=args.child_overlap,
        window_size=max(0, int(args.window_size)),
        caption_context_max_children=max(1, int(args.caption_context_max_children)),
    )
    write_jsonl(parent_chunks, args.parent_output)
    write_jsonl(child_chunks, args.child_output)
    write_jsonl(parent_index, args.parent_index_output)

    print(f"Loaded source parent chunks: {len(chunks)}")
    print(f"Wrote parent chunks: {len(parent_chunks)} -> {args.parent_output}")
    print(f"Wrote child chunks: {len(child_chunks)} -> {args.child_output}")
    print(f"Wrote parent index records: {len(parent_index)} -> {args.parent_index_output}")
    print(f"Child size/overlap: {args.child_size}/{args.child_overlap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
