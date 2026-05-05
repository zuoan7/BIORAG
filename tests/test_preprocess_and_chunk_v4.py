from __future__ import annotations

from scripts.ingestion.preprocess_and_chunk import process_document


def clean_block(
    text: str,
    block_type: str,
    order: int,
    *,
    page: int = 1,
    block_id: str | None = None,
    source_block_id: str | None = None,
) -> dict:
    bid = block_id or f"p{page}_b{order:04d}"
    sid = source_block_id or bid
    return {
        "block_id": bid,
        "type": block_type,
        "text": text,
        "section_path": ["Introduction"],
        "page": page,
        "metadata": {
            "source_block_id": sid,
            "bbox": [72.0, 100.0 + order * 10, 300.0, 108.0 + order * 10],
            "column": "L",
            "reading_order": order,
        },
    }


def make_doc(blocks: list[dict], *, doc_id: str = "doc_test") -> dict:
    text = "\n\n".join(b.get("text", "") for b in blocks if b.get("text"))
    return {
        "doc_id": doc_id,
        "source_file": f"{doc_id}.pdf",
        "parser_stage": "parsed_clean_v4",
        "has_blocks": True,
        "raw_text": text,
        "pages": [{"page": 1, "text": text, "blocks": blocks}],
    }


def chunk_doc(blocks: list[dict], **kwargs):
    doc = make_doc(blocks)
    chunks, low_quality = process_document(
        doc,
        chunk_size=80,
        chunk_overlap=10,
        min_chunk_chars=1,
        min_chunk_words=1,
        quality_threshold=0.0,
        **kwargs,
    )
    assert not low_quality
    return chunks


def test_metadata_noise_image_excluded_from_chunk_text() -> None:
    chunks = chunk_doc([
        clean_block("Introduction", "section_heading", 1),
        clean_block("This paragraph contains useful fermentation evidence.", "paragraph", 2),
        clean_block("Correspondence: author@example.org", "metadata", 3),
        clean_block("page noise", "noise", 4),
        clean_block("", "image", 5),
    ])

    text = "\n".join(c.text for c in chunks)
    assert "useful fermentation evidence" in text
    assert "Correspondence" not in text
    assert "page noise" not in text
    assert all("metadata" not in c.block_types for c in chunks)
    assert chunks[0].excluded_block_counts["metadata"] == 1
    assert chunks[0].excluded_block_counts["noise"] == 1
    assert chunks[0].excluded_block_counts["image"] == 1


def test_references_excluded_by_default() -> None:
    chunks = chunk_doc([
        clean_block("Introduction", "section_heading", 1),
        clean_block("This paragraph contains useful body evidence.", "paragraph", 2),
        clean_block("1. Smith et al. Journal 10, 1-9 (2024).", "references", 3),
    ])

    text = "\n".join(c.text for c in chunks)
    assert "Smith et al." not in text
    assert chunks[0].excluded_block_counts["references"] == 1
    assert not chunks[0].contains_references


def test_source_block_ids_and_layout_metadata_retained() -> None:
    chunks = chunk_doc([
        clean_block("Introduction", "section_heading", 1),
        clean_block("This paragraph contains useful layout evidence.", "paragraph", 2, source_block_id="raw_p1_b0002"),
    ])

    chunk = chunks[0]
    assert "raw_p1_b0002" in chunk.source_block_ids
    assert "p1_b0002" in chunk.block_ids
    assert "paragraph" in chunk.block_types
    assert chunk.layout_columns == ["L"]
    assert chunk.reading_order_span["start"] == 1
    assert chunk.bbox_span["x0"] == 72.0
    assert any(m.get("source_block_id") == "raw_p1_b0002" for m in chunk.source_block_metadata)


def test_figure_caption_retained_with_marker_and_flag() -> None:
    chunks = chunk_doc([
        clean_block("Fig. 1. Overview", "figure_caption", 1),
    ])

    assert "[FIGURE CAPTION] Fig. 1. Overview" in chunks[0].text
    assert chunks[0].contains_figure_caption is True
    assert "figure_caption" in chunks[0].evidence_types


def test_table_caption_retained_with_marker() -> None:
    chunks = chunk_doc([
        clean_block("Table S1: Primers", "table_caption", 1),
    ])

    assert "[TABLE CAPTION] Table S1: Primers" in chunks[0].text
    assert chunks[0].contains_table_caption is True


def test_table_text_retained_even_when_short() -> None:
    chunks = chunk_doc([
        clean_block("OD600 1.0 2.0", "table_text", 1),
    ])

    assert "[TABLE TEXT] OD600 1.0 2.0" in chunks[0].text
    assert chunks[0].contains_table_text is True


def test_table_caption_and_table_text_adjacency() -> None:
    chunks = chunk_doc([
        clean_block("Table 1: Strains", "table_caption", 1),
        clean_block("Strain Plasmid Product Titer", "table_text", 2),
    ])

    assert len(chunks) == 1
    assert "[TABLE CAPTION]" in chunks[0].text
    assert "[TABLE TEXT]" in chunks[0].text


def test_no_type_text_assumptions() -> None:
    chunks = chunk_doc([
        clean_block("Introduction", "section_heading", 1),
        clean_block("Standardized clean blocks are consumed without raw text type.", "paragraph", 2),
    ])

    assert all("text" not in c.block_types for c in chunks)


def test_old_format_fallback_without_blocks() -> None:
    doc = {
        "doc_id": "doc_old",
        "source_file": "doc_old.pdf",
        "raw_text": "Introduction\n\nThis old format paragraph remains chunkable.",
        "has_blocks": False,
        "pages": [{"page": 1, "text": "Introduction\n\nThis old format paragraph remains chunkable."}],
    }
    chunks, low_quality = process_document(doc, min_chunk_chars=1, min_chunk_words=1, quality_threshold=0.0)

    assert not low_quality
    assert chunks
    assert "old format paragraph" in chunks[0].text


def test_contamination_regression_excluded_from_chunk_text() -> None:
    chunks = chunk_doc([
        clean_block("This paragraph contains useful body evidence.", "paragraph", 1),
        clean_block("Journal Pre-proof", "metadata", 2),
        clean_block("*Correspondence: author@example.org", "metadata", 3),
        clean_block("at University of Hawaii at Manoa Library on June 16, 2015", "metadata", 4),
    ])

    text = "\n".join(c.text for c in chunks)
    assert "Journal Pre-proof" not in text
    assert "Correspondence" not in text
    assert "University of Hawaii" not in text
