from __future__ import annotations

from dataclasses import asdict

from scripts.ingestion.preprocess_and_chunk import process_document


def clean_block(
    text: str,
    block_type: str,
    order: int,
    *,
    page: int = 1,
    block_id: str | None = None,
    source_block_id: str | None = None,
    section_path: list[str] | None = None,
) -> dict:
    bid = block_id or f"p{page}_b{order:04d}"
    sid = source_block_id or bid
    return {
        "block_id": bid,
        "type": block_type,
        "text": text,
        "section_path": section_path or ["Introduction"],
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
    assert any(m.get("section_path") == ["Introduction"] for m in chunk.source_block_metadata)
    assert any(m.get("text_preview", "").startswith("This paragraph contains useful layout evidence.") for m in chunk.source_block_metadata)


def test_alias_fields_and_block_id_fallback_retained() -> None:
    doc = {
        "doc_id": "doc_alias",
        "source_file": "doc_alias.pdf",
        "has_blocks": True,
        "raw_text": "Alias heading\n\nAlias paragraph evidence remains chunkable.",
        "pages": [{
            "page": 1,
            "text": "Alias heading\n\nAlias paragraph evidence remains chunkable.",
            "blocks": [
                {
                    "id": "alias_h1",
                    "type": "section_heading",
                    "text": "Introduction",
                    "section_path": ["Introduction"],
                    "page_number": 1,
                    "layout_column": "R",
                    "order": 7,
                    "box": [12.0, 24.0, 48.0, 60.0],
                },
                {
                    "id": "alias_p1",
                    "type": "paragraph",
                    "text": "Alias paragraph evidence remains chunkable.",
                    "section_path": ["Introduction"],
                    "page_number": 1,
                    "layout_column": "R",
                    "order": 8,
                    "box": [12.0, 64.0, 180.0, 92.0],
                },
            ],
        }],
    }

    chunks, low_quality = process_document(
        doc,
        chunk_size=80,
        chunk_overlap=10,
        min_chunk_chars=1,
        min_chunk_words=1,
        quality_threshold=0.0,
    )

    assert not low_quality
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.parser_stage == "parsed_clean_v1"
    assert chunk.source_block_ids == ["alias_h1", "alias_p1"]
    assert chunk.block_ids == ["alias_h1", "alias_p1"]
    assert chunk.page_numbers == [1]
    assert chunk.layout_columns == ["R"]
    assert chunk.reading_order_span == {"start": 7, "end": 8}
    assert chunk.bbox_span == {"x0": 12.0, "y0": 24.0, "x1": 180.0, "y1": 92.0}


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


def test_numeric_table_text_survives_when_isolated() -> None:
    chunks = chunk_doc([
        clean_block("0.13 0.018 0.030 0.022 0.012", "table_text", 1),
    ])

    assert len(chunks) == 1
    assert chunks[0].block_types == ["table_text"]
    assert chunks[0].contains_table_text is True
    assert "[TABLE TEXT] 0.13 0.018 0.030" in chunks[0].text


def test_table_caption_and_table_text_adjacency() -> None:
    chunks = chunk_doc([
        clean_block("Table 1: Strains", "table_caption", 1),
        clean_block("Strain Plasmid Product Titer", "table_text", 2),
    ])

    assert len(chunks) == 1
    assert "[TABLE CAPTION]" in chunks[0].text
    assert "[TABLE TEXT]" in chunks[0].text


def test_figure_caption_is_separated_from_surrounding_paragraphs() -> None:
    chunks = chunk_doc([
        clean_block("Introduction", "section_heading", 1),
        clean_block("The first paragraph describes useful fermentation context.", "paragraph", 2),
        clean_block("Fig. 1. Overview of the pathway.", "figure_caption", 3),
        clean_block("The following paragraph continues the body discussion.", "paragraph", 4),
    ])

    assert len(chunks) == 3
    assert chunks[0].block_types == ["paragraph", "section_heading"]
    assert chunks[1].block_types == ["figure_caption"]
    assert chunks[2].block_types == ["paragraph"]
    assert chunks[1].contains_figure_caption is True
    assert "[FIGURE CAPTION] Fig. 1. Overview" in chunks[1].text
    assert "first paragraph" not in chunks[1].text
    assert "following paragraph" not in chunks[1].text


def test_table_caption_and_text_are_separated_from_surrounding_paragraphs() -> None:
    chunks = chunk_doc([
        clean_block("Introduction", "section_heading", 1),
        clean_block("The paragraph before the table describes the experiment.", "paragraph", 2),
        clean_block("Table 1: Strains used in this study.", "table_caption", 3),
        clean_block("Strain Plasmid Product Titer", "table_text", 4),
        clean_block("The paragraph after the table interprets the results.", "paragraph", 5),
    ])

    assert len(chunks) == 3
    assert chunks[0].block_types == ["paragraph", "section_heading"]
    assert chunks[1].block_types == ["table_caption", "table_text"]
    assert chunks[2].block_types == ["paragraph"]
    assert chunks[1].contains_table_caption is True
    assert chunks[1].contains_table_text is True
    assert "[TABLE CAPTION] Table 1: Strains" in chunks[1].text
    assert "[TABLE TEXT] Strain Plasmid Product Titer" in chunks[1].text
    assert "paragraph before" not in chunks[1].text
    assert "paragraph after" not in chunks[1].text


def test_orphan_table_text_rows_form_table_text_focused_chunk() -> None:
    chunks = chunk_doc([
        clean_block("Introduction", "section_heading", 1),
        clean_block("The paragraph before orphan rows describes measurements.", "paragraph", 2),
        clean_block("OD600 0.5 1.0 1.5", "table_text", 3),
        clean_block("Titer 10 20 30 mg/L", "table_text", 4),
        clean_block("The paragraph after orphan rows resumes body text.", "paragraph", 5),
    ])

    assert len(chunks) == 3
    assert chunks[1].block_types == ["table_text"]
    assert chunks[1].evidence_types == ["table_text"]
    assert chunks[1].contains_table_text is True
    assert chunks[1].contains_table_caption is False
    assert "[TABLE TEXT] OD600 0.5 1.0 1.5" in chunks[1].text
    assert "[TABLE TEXT] Titer 10 20 30 mg/L" in chunks[1].text
    assert "paragraph before" not in chunks[1].text
    assert "paragraph after" not in chunks[1].text


def test_table_and_figure_focused_chunks_retain_provenance_and_flags() -> None:
    chunks = chunk_doc([
        clean_block("Introduction", "section_heading", 1, page=1),
        clean_block("Table 1: Strains", "table_caption", 2, page=1, source_block_id="raw_caption"),
        clean_block("Strain Plasmid Product Titer", "table_text", 3, page=2, source_block_id="raw_table_text"),
        clean_block("Fig. 1. Pathway overview.", "figure_caption", 4, page=2, source_block_id="raw_figure"),
    ])

    table_chunk = next(chunk for chunk in chunks if chunk.contains_table_caption)
    assert table_chunk.block_types == ["table_caption", "table_text"]
    assert table_chunk.evidence_types == ["table_caption", "table_text"]
    assert table_chunk.source_block_ids == ["raw_caption", "raw_table_text"]
    assert table_chunk.block_ids == ["p1_b0002", "p2_b0003"]
    assert table_chunk.page_numbers == [1, 2]
    assert table_chunk.contains_table_caption is True
    assert table_chunk.contains_table_text is True
    assert table_chunk.contains_figure_caption is False

    figure_chunk = next(chunk for chunk in chunks if chunk.contains_figure_caption)
    assert figure_chunk.block_types == ["figure_caption"]
    assert figure_chunk.evidence_types == ["figure_caption"]
    assert figure_chunk.source_block_ids == ["raw_figure"]
    assert figure_chunk.block_ids == ["p2_b0004"]
    assert figure_chunk.page_numbers == [2]
    assert figure_chunk.contains_figure_caption is True
    assert figure_chunk.contains_table_caption is False
    assert figure_chunk.contains_table_text is False


def test_table_caption_can_join_same_table_text_across_intervening_figure_caption() -> None:
    chunks = chunk_doc([
        clean_block("Introduction", "section_heading", 1),
        clean_block("Table 1: Strains", "table_caption", 2),
        clean_block("Fig. 1. Pathway overview.", "figure_caption", 3),
        clean_block("Strain Plasmid Product Titer", "table_text", 4),
    ])

    table_chunk = next(chunk for chunk in chunks if chunk.contains_table_caption)
    figure_chunk = next(chunk for chunk in chunks if chunk.contains_figure_caption)
    assert table_chunk.block_types == ["table_caption", "table_text"]
    assert "[TABLE CAPTION] Table 1: Strains" in table_chunk.text
    assert "[TABLE TEXT] Strain Plasmid Product Titer" in table_chunk.text
    assert "FIGURE CAPTION" not in table_chunk.text
    assert figure_chunk.block_types == ["figure_caption"]


def test_evidence_chunk_section_prefers_non_title_section_path() -> None:
    chunks = chunk_doc([
        clean_block(
            "Fig. 1. Pathway overview.",
            "figure_caption",
            1,
            section_path=["Title", "2.1. Materials"],
        ),
    ])

    assert chunks[0].section == "2.1. Materials"
    assert "section: 2.1. Materials" in chunks[0].retrieval_text


def test_paragraph_only_chunking_remains_aggregated() -> None:
    chunks = chunk_doc([
        clean_block("First paragraph contains ordinary body evidence.", "paragraph", 1),
        clean_block("Second paragraph continues ordinary body evidence.", "paragraph", 2),
        clean_block("Third paragraph remains in the same ordinary flow.", "paragraph", 3),
    ])

    assert len(chunks) == 1
    assert chunks[0].block_types == ["paragraph"]
    assert chunks[0].contains_table_caption is False
    assert chunks[0].contains_table_text is False
    assert chunks[0].contains_figure_caption is False
    assert "First paragraph" in chunks[0].text
    assert "Third paragraph" in chunks[0].text


def test_evidence_context_mode_off_leaves_paragraph_retrieval_text_unchanged() -> None:
    blocks = [
        clean_block("First paragraph contains ordinary body evidence.", "paragraph", 1),
        clean_block("Second paragraph continues ordinary body evidence.", "paragraph", 2),
    ]

    default_chunks = chunk_doc(blocks)
    off_chunks = chunk_doc(blocks, evidence_context_mode="off")

    assert len(default_chunks) == len(off_chunks) == 1
    assert default_chunks[0].text == off_chunks[0].text
    assert default_chunks[0].retrieval_text == off_chunks[0].retrieval_text
    assert "[Table Evidence]" not in off_chunks[0].retrieval_text
    assert "[Figure Evidence]" not in off_chunks[0].retrieval_text


def test_compact_evidence_context_enriches_table_and_figure_only() -> None:
    blocks = [
        clean_block("Introduction", "section_heading", 1, page=1),
        clean_block("The paragraph before the table describes the experiment.", "paragraph", 2, page=1),
        clean_block("Table 1: Strains used in this study.", "table_caption", 3, page=2),
        clean_block("Strain Plasmid Product Titer", "table_text", 4, page=2),
        clean_block("Fig. 1. Pathway overview.", "figure_caption", 5, page=3),
        clean_block("The paragraph after the figure resumes body text.", "paragraph", 6, page=3),
    ]

    off_chunks = chunk_doc(blocks, evidence_context_mode="off")
    compact_chunks = chunk_doc(blocks, evidence_context_mode="compact")

    assert [chunk.text for chunk in compact_chunks] == [chunk.text for chunk in off_chunks]
    assert {tuple(asdict(chunk).keys()) for chunk in compact_chunks} == {
        tuple(asdict(chunk).keys()) for chunk in off_chunks
    }

    table_chunk = next(chunk for chunk in compact_chunks if chunk.contains_table_caption)
    assert table_chunk.retrieval_text.startswith("[Table Evidence]\n")
    assert "Section: Introduction" in table_chunk.retrieval_text
    assert "Page: 2" in table_chunk.retrieval_text
    assert "[TABLE CAPTION] Table 1: Strains" in table_chunk.retrieval_text
    assert "[TABLE TEXT] Strain Plasmid Product Titer" in table_chunk.retrieval_text

    figure_chunk = next(chunk for chunk in compact_chunks if chunk.contains_figure_caption)
    assert figure_chunk.retrieval_text.startswith("[Figure Evidence]\n")
    assert "Section: Introduction" in figure_chunk.retrieval_text
    assert "Page: 3" in figure_chunk.retrieval_text
    assert "[FIGURE CAPTION] Fig. 1. Pathway overview." in figure_chunk.retrieval_text

    paragraph_chunks = [
        chunk for chunk in compact_chunks
        if not (chunk.contains_table_caption or chunk.contains_table_text or chunk.contains_figure_caption)
    ]
    assert paragraph_chunks
    assert all("[Table Evidence]" not in chunk.retrieval_text for chunk in paragraph_chunks)
    assert all("[Figure Evidence]" not in chunk.retrieval_text for chunk in paragraph_chunks)


def test_compact_context_preserves_supporting_information_figure_caption() -> None:
    chunks = chunk_doc([
        clean_block(
            "Figure 5. Comparison of Opto-T7RNAPs "
            "(Supporting Information: paT7P-1 Characterization).",
            "figure_caption",
            1,
        ),
    ], evidence_context_mode="compact")

    assert len(chunks) == 1
    assert chunks[0].contains_figure_caption is True
    assert "[Figure Evidence]" in chunks[0].retrieval_text
    assert "Supporting Information: paT7P-1 Characterization" in chunks[0].text
    assert "Supporting Information: paT7P-1 Characterization" in chunks[0].retrieval_text


def test_compact_context_keeps_numeric_only_table_text() -> None:
    chunks = chunk_doc([
        clean_block("0.13 0.018 0.030 0.022 0.012", "table_text", 1),
    ], evidence_context_mode="compact")

    assert len(chunks) == 1
    assert chunks[0].contains_table_text is True
    assert chunks[0].text.startswith("[TABLE TEXT] 0.13 0.018")
    assert chunks[0].retrieval_text.startswith("[Table Evidence]\n")
    assert "[TABLE TEXT] 0.13 0.018" in chunks[0].retrieval_text


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


def test_chunk_defense_excludes_cover_header_and_annotation_contamination() -> None:
    chunks = chunk_doc([
        clean_block("Useful fermentation body evidence remains.", "paragraph", 1),
        clean_block("S1096-7176(25)00185-5", "paragraph", 2),
        clean_block("YMBEN 2419", "paragraph", 3),
        clean_block("This is a PDF of an article that has undergone enhancements after acceptance", "paragraph", 4),
        clean_block("Page 2 of 14", "paragraph", 5),
        clean_block("OPEN ACCESS", "paragraph", 6),
        clean_block("表达 Fam20C 失败,是", "paragraph", 7),
    ])

    text = "\n".join(c.text for c in chunks)
    assert "Useful fermentation body evidence" in text
    assert "S1096-7176" not in text
    assert "YMBEN" not in text
    assert "This is a PDF of an article" not in text
    assert "Page 2 of 14" not in text
    assert "OPEN ACCESS" not in text
    assert "Fam20C" not in text


def test_chunk_defense_excludes_cover_disclaimer_and_credit_metadata() -> None:
    chunks = chunk_doc([
        clean_block("Useful body evidence remains.", "paragraph", 1),
        clean_block("Metabolic Engineering", "paragraph", 2),
        clean_block("23 November 2023", "paragraph", 3),
        clean_block("in its final form, but we are providing this version to give early visibility of the article. Please note that,", "paragraph", 4),
        clean_block("disclaimers that apply to the journal pertain.", "paragraph", 5),
        clean_block("Shun Endo: Investigation (lead, equal). Sayaka Kamai: Investigation (lead, equal).", "paragraph", 6),
        clean_block("Formal analysis; Writing - original draft. Kento Koketsu: Supervision; Writing - review & editing.", "paragraph", 7),
    ])

    text = "\n".join(c.text for c in chunks)
    assert "Useful body evidence" in text
    assert "Metabolic Engineering" not in text
    assert "23 November 2023" not in text
    assert "early visibility" not in text
    assert "Investigation (lead" not in text
    assert "Writing - review" not in text


def test_chunk_defense_demotes_false_table_text_marker() -> None:
    chunks = chunk_doc([
        clean_block("After identification of the sequence encoding active Hac1p we evaluated the effect.", "table_text", 1),
        clean_block("Primer Sequence OD600 1.0 2.0 3.0", "table_text", 2),
    ])

    text = "\n".join(c.text for c in chunks)
    assert "[TABLE TEXT] After identification" not in text
    assert "After identification of the sequence" in text
    assert "[TABLE TEXT] Primer Sequence OD600" in text


def test_chunk_defense_demotes_protocol_recipe_table_text() -> None:
    chunks = chunk_doc([
        clean_block("gradient from 15 to 80% acetonitrile in 0.1% formic acid", "table_text", 1),
        clean_block("umn, 100 Å, 1.8 μm, 300 μm × 150 mm, Waters) using", "table_text", 2),
        clean_block("from 3.5% B (B: 80% ACN, 20% A) to 40% B in 30 min", "table_text", 3),
        clean_block("tryptone, 1 g yeast extract, 34 g NaCl, and 0.1 g FePO4,", "table_text", 4),
        clean_block("(NH4)2SO4, 1 mM MgSO4, 3.9 μM FeSO4, and 1 g (g/l)", "table_text", 5),
    ])

    text = "\n".join(c.text for c in chunks)
    assert "gradient from 15 to 80% acetonitrile" in text
    assert "tryptone, 1 g yeast extract" in text
    assert "[TABLE TEXT] gradient" not in text
    assert "[TABLE TEXT] umn," not in text
    assert "[TABLE TEXT] from 3.5% B" not in text
    assert "[TABLE TEXT] tryptone" not in text
    assert "[TABLE TEXT] (NH4)2SO4" not in text


def test_doc0005_like_table_rows_reach_chunk_text() -> None:
    chunks = chunk_doc([
        clean_block("Table 3 continued", "table_caption", 1),
        clean_block("Orf_name", "table_text", 2),
        clean_block("Length (bp)", "table_text", 3),
        clean_block("Encoding protein", "table_text", 4),
        clean_block("gm_orf2729", "table_text", 5),
        clean_block("369", "table_text", 6),
        clean_block("Cytochrome c551", "table_text", 7),
    ])

    text = "\n".join(c.text for c in chunks)
    assert "[TABLE CAPTION] Table 3 continued" in text
    assert "[TABLE TEXT] gm_orf2729" in text
    assert "[TABLE TEXT] Cytochrome c551" in text


def test_page_header_not_in_chunk_text() -> None:
    chunks = chunk_doc([
        clean_block("Zhu et al. Biotechnol Biofuels (2017) 10:44", "metadata", 1),
        clean_block("Page 11 of 14", "metadata", 2),
        clean_block("Table 3 continued", "table_caption", 3),
        clean_block("gm_orf2729", "table_text", 4),
    ])

    text = "\n".join(c.text for c in chunks)
    assert "gm_orf2729" in text
    assert "Page 11 of 14" not in text
    assert "Zhu et al. Biotechnol Biofuels" not in text


def test_fig6_caption_retained() -> None:
    chunks = chunk_doc([
        clean_block("Fig. 6 Putative lignin degradation pathways of strain L1.", "figure_caption", 1),
    ])

    assert "[FIGURE CAPTION] Fig. 6 Putative lignin degradation pathways" in chunks[0].text
    assert chunks[0].contains_figure_caption is True


def test_fallback_fulltext_preserves_figure_caption_with_supporting_information() -> None:
    doc = make_doc([
        clean_block("Optogenetic control of T7 RNA polymerase", "title", 1, section_path=["Title"]),
        clean_block("The paper describes a light-inducible transcription system.", "paragraph", 2, section_path=["Title"]),
        clean_block(
            "Figure 5. Comparison of Opto-T7RNAPs to paT7P-148 "
            "(Supporting Information: paT7P-1 Characterization).",
            "figure_caption",
            3,
            section_path=["Title"],
        ),
        clean_block("The following paragraph interprets the figure.", "paragraph", 4, section_path=["Title"]),
    ], doc_id="doc_fallback_figure")

    chunks, low_quality = process_document(
        doc,
        chunk_size=80,
        chunk_overlap=10,
        min_chunk_chars=1,
        min_chunk_words=1,
        quality_threshold=0.0,
    )

    assert not low_quality
    figure_chunk = next(chunk for chunk in chunks if chunk.contains_figure_caption)
    assert "p1_b0003" in figure_chunk.block_ids
    assert "[FIGURE CAPTION] Figure 5. Comparison of Opto-T7RNAPs" in figure_chunk.text
