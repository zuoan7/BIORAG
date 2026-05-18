from __future__ import annotations

from collections import Counter

from ..domain.schemas import RetrievedChunk


SUMMARY_SECTIONS = {"abstract", "conclusion", "conclusions"}


def build_empty_supplement_debug() -> dict:
    return {
        "enabled": False,
        "used": False,
        "reason": "",
        "doc_ids": [],
        "chunk_ids": [],
        "sections": [],
        "count": 0,
        "source": "",
        "abstract_or_conclusion_available_count": 0,
        "abstract_or_conclusion_added_count": 0,
    }


def supplement_summary_sections(
    *,
    question: str,
    seed_chunks: list[RetrievedChunk],
    milvus_client,
    collection_name: str,
    max_docs: int = 3,
    max_per_doc: int = 2,
    max_total: int = 5,
) -> tuple[list[RetrievedChunk], dict]:
    """Supplement summary seed chunks with Abstract/Conclusion chunks."""
    if milvus_client is None:
        return seed_chunks, build_empty_supplement_debug()

    doc_counts: Counter[str] = Counter()
    for chunk in seed_chunks:
        if chunk.doc_id:
            doc_counts[chunk.doc_id] += 1
    top_docs = [doc for doc, _ in doc_counts.most_common(max_docs)]

    existing_abs_conc = set()
    for chunk in seed_chunks:
        section_lower = (chunk.section or "").lower()
        if section_lower in SUMMARY_SECTIONS and chunk.doc_id in top_docs:
            existing_abs_conc.add(chunk.doc_id)

    missing_docs = [d for d in top_docs if d not in existing_abs_conc]
    if not missing_docs:
        return seed_chunks, build_empty_supplement_debug()

    supplemental_chunks: list[RetrievedChunk] = []
    added_doc_ids: list[str] = []
    added_chunk_ids: list[str] = []
    added_sections: list[str] = []
    abstract_conc_available = 0

    for doc_id in missing_docs[:max_docs]:
        if len(supplemental_chunks) >= max_total:
            break
        doc_supplement_count = 0
        for section in ("Abstract", "Conclusion", "Conclusions"):
            if doc_supplement_count >= max_per_doc or len(supplemental_chunks) >= max_total:
                break
            filter_expr = f'doc_id == "{doc_id}" and section == "{section}"'
            try:
                results = milvus_client.query(
                    collection_name=collection_name,
                    filter=filter_expr,
                    output_fields=[
                        "chunk_id", "doc_id", "source_file", "title",
                        "section", "page_start", "page_end", "chunk_index", "text",
                    ],
                    limit=2,
                )
            except Exception:
                continue

            for hit in (results or []):
                text = hit.get("text") or ""
                if len(text) < 20:
                    continue
                if _is_bibliography_like(text):
                    continue
                abstract_conc_available += 1

                chunk_id = hit.get("chunk_id", "")
                if any(c.chunk_id == chunk_id for c in seed_chunks):
                    continue

                chunk = RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=hit.get("doc_id", ""),
                    source_file=hit.get("source_file", ""),
                    title=hit.get("title", ""),
                    section=hit.get("section", ""),
                    text=text,
                    page_start=hit.get("page_start"),
                    page_end=hit.get("page_end"),
                    vector_score=0.0,
                    bm25_score=0.0,
                    rerank_score=None,
                    fusion_score=None,
                    metadata={"chunk_index": hit.get("chunk_index")},
                )
                supplemental_chunks.append(chunk)
                added_doc_ids.append(doc_id)
                added_chunk_ids.append(chunk_id)
                added_sections.append(hit.get("section", ""))
                doc_supplement_count += 1

    if not supplemental_chunks:
        debug = {
            "enabled": True,
            "used": False,
            "reason": f"no_abstract_conclusion_found_for_missing_docs:{','.join(missing_docs[:3])}",
            "doc_ids": missing_docs[:3],
            "chunk_ids": [],
            "sections": [],
            "count": 0,
            "source": "retrieved_doc",
            "abstract_or_conclusion_available_count": abstract_conc_available,
            "abstract_or_conclusion_added_count": 0,
        }
        return seed_chunks, debug

    all_chunks = list(seed_chunks) + supplemental_chunks
    debug = {
        "enabled": True,
        "used": True,
        "reason": f"supplemented_abstract_conclusion_from_{len(missing_docs)}_docs",
        "doc_ids": added_doc_ids,
        "chunk_ids": added_chunk_ids,
        "sections": added_sections,
        "count": len(supplemental_chunks),
        "source": "retrieved_doc",
        "abstract_or_conclusion_available_count": abstract_conc_available,
        "abstract_or_conclusion_added_count": len(supplemental_chunks),
    }
    return all_chunks, debug


def _is_bibliography_like(text: str) -> bool:
    lowered = text.lower()
    doi_count = lowered.count("https://doi.org")
    if doi_count >= 2:
        return True
    et_al_count = lowered.count("et al.")
    if et_al_count >= 3:
        return True
    return False
