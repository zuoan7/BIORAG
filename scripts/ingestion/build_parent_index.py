from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from hashlib import sha1
from pathlib import Path
from typing import Any


def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for ordinal, raw in enumerate(handle):
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            item["_ordinal"] = ordinal
            chunks.append(item)
    return chunks


def build_parent_records(
    chunks: list[dict[str, Any]],
    window_size: int = 1,
    caption_context_max_children: int = 5,
) -> list[dict[str, Any]]:
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        doc_id = str(chunk.get("doc_id") or "")
        if not doc_id:
            continue
        by_doc[doc_id].append(chunk)

    parents: list[dict[str, Any]] = []
    for doc_id, doc_chunks in by_doc.items():
        ordered = sorted(doc_chunks, key=_chunk_sort_key)
        parents.append(_build_doc_parent(doc_id, ordered))

        section_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        section_path_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        page_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        evidence_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for idx, chunk in enumerate(ordered):
            section_groups[str(chunk.get("section") or "")].append(chunk)

            section_path_key = _section_path_key_for_chunk(chunk)
            section_path_groups[section_path_key].append(chunk)

            for page_number in _coerce_int_list(chunk.get("page_numbers")):
                page_groups[page_number].append(chunk)

            for evidence_type in _infer_evidence_types(chunk):
                evidence_groups[evidence_type].append(chunk)

            window_chunks = ordered[max(0, idx - window_size): idx + window_size + 1]
            parents.append(_build_window_parent(doc_id, chunk, window_chunks, window_size))
            if _is_caption_chunk(chunk):
                caption_chunks = _select_caption_context_chunks(
                    ordered=ordered,
                    anchor_index=idx,
                    anchor_chunk=chunk,
                    window_size=window_size,
                    max_children=caption_context_max_children,
                )
                parents.append(_build_caption_parent(doc_id, chunk, caption_chunks, window_size))

        for section_name, section_chunks in section_groups.items():
            parents.append(_build_section_parent(doc_id, section_name, section_chunks))

        for section_path_key, section_chunks in section_path_groups.items():
            parents.append(_build_section_path_parent(doc_id, section_path_key, section_chunks))

        for page_number, page_chunks in page_groups.items():
            parents.append(_build_page_parent(doc_id, page_number, page_chunks))

        for evidence_type, evidence_chunks in evidence_groups.items():
            parents.append(_build_evidence_parent(doc_id, evidence_type, evidence_chunks))

    return parents


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build parent index sidecar from chunk JSONL.")
    parser.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--output", required=True, help="Path to parent_index.jsonl")
    parser.add_argument("--window-size", type=int, default=1, help="Neighbor window size for window/caption parents")
    parser.add_argument(
        "--caption-context-max-children",
        type=int,
        default=5,
        help="Maximum children retained in each caption_context parent",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    parents = build_parent_records(
        chunks,
        window_size=max(0, int(args.window_size)),
        caption_context_max_children=max(1, int(args.caption_context_max_children)),
    )
    write_jsonl(parents, args.output)

    parent_types = Counter(record["parent_type"] for record in parents)
    print(f"Loaded chunks: {len(chunks)}")
    print(f"Wrote parents: {len(parents)} -> {args.output}")
    print(f"Parent types: {dict(parent_types)}")
    return 0


def make_parent_id(parent_type: str, doc_id: str, key: str) -> str:
    suffix = sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{doc_id}::{parent_type}::{suffix}"


def _build_doc_parent(doc_id: str, doc_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    first = doc_chunks[0] if doc_chunks else {}
    return _make_parent_record(
        parent_id=make_parent_id("doc", doc_id, f"{doc_id}::doc"),
        parent_type="doc",
        doc_id=doc_id,
        source_file=str(first.get("source_file") or ""),
        title=str(first.get("title") or ""),
        section="",
        section_path=[],
        section_path_key="",
        anchor_chunk_id="",
        child_chunks=doc_chunks,
        caption_kind="",
        evidence_type="",
        page_number=None,
    )


def _build_section_parent(doc_id: str, section_name: str, section_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    first = section_chunks[0] if section_chunks else {}
    section_path = _coerce_str_list(first.get("section_path"))
    return _make_parent_record(
        parent_id=make_parent_id("section", doc_id, f"{doc_id}::section::{section_name}"),
        parent_type="section",
        doc_id=doc_id,
        source_file=str(first.get("source_file") or ""),
        title=str(first.get("title") or ""),
        section=section_name,
        section_path=section_path,
        section_path_key=_normalize_section_path(section_path) if section_path else "",
        anchor_chunk_id="",
        child_chunks=section_chunks,
        caption_kind="",
        evidence_type="",
        page_number=None,
    )


def _build_section_path_parent(doc_id: str, section_path_key: str, section_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    first = section_chunks[0] if section_chunks else {}
    section_path = _coerce_str_list(first.get("section_path"))
    if not section_path:
        section_name = str(first.get("section") or "")
        section_path = [section_name] if section_name else []
    return _make_parent_record(
        parent_id=make_parent_id("section_path", doc_id, f"{doc_id}::section_path::{section_path_key}"),
        parent_type="section_path",
        doc_id=doc_id,
        source_file=str(first.get("source_file") or ""),
        title=str(first.get("title") or ""),
        section=str(first.get("section") or ""),
        section_path=section_path,
        section_path_key=section_path_key,
        anchor_chunk_id="",
        child_chunks=section_chunks,
        caption_kind="",
        evidence_type="",
        page_number=None,
    )


def _build_page_parent(doc_id: str, page_number: int, page_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    first = page_chunks[0] if page_chunks else {}
    return _make_parent_record(
        parent_id=f"{doc_id}::page::{page_number}",
        parent_type="page",
        doc_id=doc_id,
        source_file=str(first.get("source_file") or ""),
        title=str(first.get("title") or ""),
        section=str(first.get("section") or ""),
        section_path=_coerce_str_list(first.get("section_path")),
        section_path_key=_section_path_key_for_chunk(first),
        anchor_chunk_id="",
        child_chunks=page_chunks,
        caption_kind="",
        evidence_type="",
        page_number=page_number,
    )


def _build_evidence_parent(doc_id: str, evidence_type: str, evidence_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    first = evidence_chunks[0] if evidence_chunks else {}
    return _make_parent_record(
        parent_id=make_parent_id(
            "evidence_type_context",
            doc_id,
            f"{doc_id}::evidence_type_context::{evidence_type}",
        ),
        parent_type="evidence_type_context",
        doc_id=doc_id,
        source_file=str(first.get("source_file") or ""),
        title=str(first.get("title") or ""),
        section=str(first.get("section") or ""),
        section_path=_coerce_str_list(first.get("section_path")),
        section_path_key=_section_path_key_for_chunk(first),
        anchor_chunk_id="",
        child_chunks=evidence_chunks,
        caption_kind="",
        evidence_type=evidence_type,
        page_number=None,
    )


def _build_window_parent(
    doc_id: str,
    anchor_chunk: dict[str, Any],
    child_chunks: list[dict[str, Any]],
    window_size: int,
) -> dict[str, Any]:
    anchor_chunk_id = str(anchor_chunk.get("chunk_id") or "")
    return _make_parent_record(
        parent_id=make_parent_id(
            "chunk_window",
            doc_id,
            f"{doc_id}::chunk_window::{anchor_chunk_id}::{window_size}",
        ),
        parent_type="chunk_window",
        doc_id=doc_id,
        source_file=str(anchor_chunk.get("source_file") or ""),
        title=str(anchor_chunk.get("title") or ""),
        section=str(anchor_chunk.get("section") or ""),
        section_path=_coerce_str_list(anchor_chunk.get("section_path")),
        section_path_key=_section_path_key_for_chunk(anchor_chunk),
        anchor_chunk_id=anchor_chunk_id,
        child_chunks=child_chunks,
        caption_kind="",
        evidence_type="",
        page_number=None,
    )


def _build_caption_parent(
    doc_id: str,
    anchor_chunk: dict[str, Any],
    child_chunks: list[dict[str, Any]],
    window_size: int,
) -> dict[str, Any]:
    anchor_chunk_id = str(anchor_chunk.get("chunk_id") or "")
    return _make_parent_record(
        parent_id=make_parent_id(
            "caption_context",
            doc_id,
            f"{doc_id}::caption_context::{anchor_chunk_id}::{window_size}",
        ),
        parent_type="caption_context",
        doc_id=doc_id,
        source_file=str(anchor_chunk.get("source_file") or ""),
        title=str(anchor_chunk.get("title") or ""),
        section=str(anchor_chunk.get("section") or ""),
        section_path=_coerce_str_list(anchor_chunk.get("section_path")),
        section_path_key=_section_path_key_for_chunk(anchor_chunk),
        anchor_chunk_id=anchor_chunk_id,
        child_chunks=child_chunks,
        caption_kind=_caption_kind(anchor_chunk),
        evidence_type="",
        page_number=None,
    )


def _make_parent_record(
    *,
    parent_id: str,
    parent_type: str,
    doc_id: str,
    source_file: str,
    title: str,
    section: str,
    section_path: list[str],
    section_path_key: str,
    anchor_chunk_id: str,
    child_chunks: list[dict[str, Any]],
    caption_kind: str,
    evidence_type: str,
    page_number: int | None,
) -> dict[str, Any]:
    ordered_children = sorted(child_chunks, key=_chunk_sort_key)
    child_chunk_ids = _ordered_unique(str(chunk.get("chunk_id") or "") for chunk in ordered_children)
    block_ids = _ordered_unique(
        block_id
        for chunk in ordered_children
        for block_id in _coerce_str_list(chunk.get("block_ids"))
    )
    source_block_ids = _ordered_unique(
        block_id
        for chunk in ordered_children
        for block_id in _coerce_str_list(chunk.get("source_block_ids"))
    )
    page_numbers = _ordered_unique_ints(
        page
        for chunk in ordered_children
        for page in _coerce_int_list(chunk.get("page_numbers"))
    )
    page_start = min(page_numbers) if page_numbers else None
    page_end = max(page_numbers) if page_numbers else None
    content_kinds = _ordered_unique(
        kind
        for chunk in ordered_children
        for kind in _derive_content_kinds(chunk)
    )
    contains_table_caption = any(bool(chunk.get("contains_table_caption")) for chunk in ordered_children)
    contains_figure_caption = any(bool(chunk.get("contains_figure_caption")) for chunk in ordered_children)
    contains_table_text = any(bool(chunk.get("contains_table_text")) for chunk in ordered_children)
    contains_image = any(bool(chunk.get("contains_image")) for chunk in ordered_children)
    text_preview = _build_text_preview(anchor_chunk_id, ordered_children)
    return {
        "parent_id": parent_id,
        "parent_type": parent_type,
        "doc_id": doc_id,
        "source_file": source_file,
        "title": title,
        "section": section,
        "section_path": section_path,
        "section_path_key": section_path_key,
        "anchor_chunk_id": anchor_chunk_id,
        "child_chunk_ids": child_chunk_ids,
        "page_start": page_start,
        "page_end": page_end,
        "page_numbers": page_numbers if page_number is None else [page_number],
        "page_number": page_number,
        "block_ids": block_ids,
        "source_block_ids": source_block_ids,
        "content_kinds": content_kinds,
        "contains_table_caption": contains_table_caption,
        "contains_figure_caption": contains_figure_caption,
        "contains_table_text": contains_table_text,
        "contains_image": contains_image,
        "text_preview": text_preview,
        "caption_kind": caption_kind,
        "evidence_type": evidence_type,
    }


def _select_caption_context_chunks(
    *,
    ordered: list[dict[str, Any]],
    anchor_index: int,
    anchor_chunk: dict[str, Any],
    window_size: int,
    max_children: int,
) -> list[dict[str, Any]]:
    anchor_pages = set(_coerce_int_list(anchor_chunk.get("page_numbers")))
    anchor_chunk_id = str(anchor_chunk.get("chunk_id") or "")
    selected: list[dict[str, Any]] = [anchor_chunk]
    seen = {anchor_chunk_id}

    same_page_candidates: list[dict[str, Any]] = []
    for chunk in ordered:
        chunk_id = str(chunk.get("chunk_id") or "")
        if not chunk_id or chunk_id in seen:
            continue
        pages = set(_coerce_int_list(chunk.get("page_numbers")))
        if anchor_pages and pages & anchor_pages:
            same_page_candidates.append(chunk)

    same_page_candidates.sort(key=_chunk_sort_key)
    for chunk in same_page_candidates:
        if len(selected) >= max_children:
            break
        chunk_id = str(chunk.get("chunk_id") or "")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        selected.append(chunk)

    for distance in range(1, max(0, window_size) + 1):
        if len(selected) >= max_children:
            break
        for idx in (anchor_index - distance, anchor_index + distance):
            if idx < 0 or idx >= len(ordered):
                continue
            chunk = ordered[idx]
            chunk_id = str(chunk.get("chunk_id") or "")
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            selected.append(chunk)
            if len(selected) >= max_children:
                break

    return sorted(selected, key=_chunk_sort_key)


def _build_text_preview(anchor_chunk_id: str, child_chunks: list[dict[str, Any]], limit: int = 280) -> str:
    ordered_texts: list[str] = []
    if anchor_chunk_id:
        for chunk in child_chunks:
            if str(chunk.get("chunk_id") or "") == anchor_chunk_id:
                ordered_texts.append(str(chunk.get("text") or ""))
                break
    ordered_texts.extend(
        str(chunk.get("text") or "")
        for chunk in child_chunks
        if str(chunk.get("chunk_id") or "") != anchor_chunk_id
    )
    compact = " ".join(part.strip() for part in ordered_texts if part and part.strip())
    compact = " ".join(compact.split())
    return compact[:limit]


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
    if chunk.get("contains_noise"):
        kinds.append("noise")
    if not kinds:
        kinds.append("body")
    return kinds


def _infer_evidence_types(chunk: dict[str, Any]) -> list[str]:
    explicit = _ordered_unique(str(v).strip() for v in chunk.get("evidence_types") or [] if str(v).strip())
    if explicit:
        return explicit

    inferred: list[str] = []
    if chunk.get("contains_table_caption") or chunk.get("contains_table_text"):
        inferred.append("table")
    if chunk.get("contains_figure_caption"):
        inferred.append("figure")

    section = str(chunk.get("section") or "").lower()
    if "method" in section or "material" in section:
        inferred.append("method")
    if "result" in section or "discussion" in section:
        inferred.append("result")

    return _ordered_unique(inferred or ["body"])


def _caption_kind(chunk: dict[str, Any]) -> str:
    has_table = bool(chunk.get("contains_table_caption"))
    has_figure = bool(chunk.get("contains_figure_caption"))
    if has_table and has_figure:
        return "mixed"
    if has_table:
        return "table_caption"
    if has_figure:
        return "figure_caption"
    return ""


def _section_path_key_for_chunk(chunk: dict[str, Any]) -> str:
    section_path = _coerce_str_list(chunk.get("section_path"))
    if section_path:
        return _normalize_section_path(section_path)
    section = str(chunk.get("section") or "").strip()
    return _normalize_section_path([section]) if section else ""


def _normalize_section_path(section_path: list[str]) -> str:
    return " > ".join(part.strip() for part in section_path if part and part.strip())


def _is_caption_chunk(chunk: dict[str, Any]) -> bool:
    return bool(chunk.get("contains_table_caption")) or bool(chunk.get("contains_figure_caption"))


def _chunk_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    return (_safe_int(item.get("chunk_index"), item.get("_ordinal", 0)), str(item.get("chunk_id") or ""))


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if str(v or "").strip()]


def _coerce_int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    output: list[int] = []
    for item in value:
        try:
            output.append(int(item))
        except (TypeError, ValueError):
            continue
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
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
