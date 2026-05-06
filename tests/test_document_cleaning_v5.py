from __future__ import annotations

from scripts.ingestion.document_cleaning_v5 import build_evidence_pack, evidence_pack_to_pages
from scripts.ingestion.preprocess_and_chunk import process_document


def clean_block(
    text: str,
    block_type: str,
    order: int,
    *,
    page: int = 1,
    block_id: str | None = None,
    metadata_extra: dict | None = None,
) -> dict:
    bid = block_id or f"p{page}_b{order:04d}"
    metadata = {
        "source_block_id": f"raw_{bid}",
        "bbox": [72.0, 100.0 + order * 10, 300.0, 108.0 + order * 10],
        "column": "L",
        "reading_order": order,
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    return {
        "block_id": bid,
        "type": block_type,
        "text": text,
        "section_path": ["Results"],
        "page": page,
        "metadata": metadata,
    }


def make_clean_doc(blocks: list[dict]) -> dict:
    return {
        "doc_id": "doc_test",
        "source_file": "doc_test.pdf",
        "total_pages": 1,
        "parser_stage": "parsed_clean_v4",
        "pages": [{"page": 1, "text": "", "blocks": blocks}],
    }


def test_evidence_pack_excludes_non_evidence_blocks() -> None:
    pack = build_evidence_pack(make_clean_doc([
        clean_block("Useful body evidence.", "paragraph", 1),
        clean_block("Correspondence: author@example.org", "metadata", 2),
        clean_block("page noise", "noise", 3),
        clean_block("", "image", 4),
        clean_block("1. Smith et al. Journal 10, 1-9 (2024).", "references", 5),
    ]))

    assert [unit["type"] for unit in pack["evidence_units"]] == ["paragraph"]
    assert pack["excluded_block_counts"]["metadata"] == 1
    assert pack["excluded_block_counts"]["noise"] == 1
    assert pack["excluded_block_counts"]["image"] == 1
    assert pack["excluded_block_counts"]["references"] == 1
    assert pack["validation_summary"]["forbidden_evidence_unit_count"] == 0


def test_evidence_pack_audits_image_metadata_without_evidence_text() -> None:
    pack = build_evidence_pack(make_clean_doc([
        clean_block(
            "",
            "image",
            1,
            metadata_extra={
                "xref": 12,
                "image_width": 640,
                "image_height": 480,
                "image_source": "page_get_images",
            },
        ),
        clean_block("Useful body evidence.", "paragraph", 2),
    ]))

    assert [unit["type"] for unit in pack["evidence_units"]] == ["paragraph"]
    assert pack["excluded_block_counts"]["image"] == 1
    assert len(pack["excluded_image_blocks"]) == 1
    image_audit = pack["excluded_image_blocks"][0]
    assert image_audit["source_block_id"] == "raw_p1_b0001"
    assert image_audit["layout"]["bbox"] == [72.0, 110.0, 300.0, 118.0]
    assert image_audit["image_metadata"]["xref"] == 12
    assert image_audit["image_metadata"]["image_width"] == 640


def test_evidence_pack_retains_layout_and_source_metadata() -> None:
    pack = build_evidence_pack(make_clean_doc([
        clean_block("Useful layout evidence.", "paragraph", 3, block_id="p1_b0003"),
    ]))

    unit = pack["evidence_units"][0]
    assert unit["source_block_id"] == "raw_p1_b0003"
    assert unit["bbox"] == [72.0, 130.0, 300.0, 138.0]
    assert unit["column"] == "L"
    assert unit["reading_order"] == 3
    assert pack["validation_summary"]["source_block_id_retention_rate"] == 1.0
    assert pack["validation_summary"]["layout_metadata_retention_rate"] == 1.0


def test_caption_and_table_text_are_retained_and_grouped() -> None:
    pack = build_evidence_pack(make_clean_doc([
        clean_block("Fig. 1. Overview", "figure_caption", 1),
        clean_block("Table 1: Strains", "table_caption", 2),
        clean_block("Strain Plasmid Product Titer", "table_text", 3),
    ]))

    types = [unit["type"] for unit in pack["evidence_units"]]
    assert types == ["figure_caption", "table_caption", "table_text"]
    table_caption = pack["evidence_units"][1]
    table_text = pack["evidence_units"][2]
    assert table_caption["table_group_id"]
    assert table_caption["table_group_id"] == table_text["table_group_id"]
    assert pack["validation_summary"]["evidence_type_counts"]["figure_caption"] == 1
    assert pack["validation_summary"]["evidence_type_counts"]["table_caption"] == 1
    assert pack["validation_summary"]["evidence_type_counts"]["table_text"] == 1


def test_false_table_text_body_sentence_is_demoted_to_paragraph_evidence() -> None:
    text = "at a compound annual growth rate of 6.3%, reaching US$4.9 billion by 2033 [7]. Caseins are"
    pack = build_evidence_pack(make_clean_doc([
        clean_block(text, "table_text", 1),
    ]))

    unit = pack["evidence_units"][0]
    assert unit["type"] == "paragraph"
    assert unit["metadata"]["source_clean_block_type"] == "table_text"
    assert unit["metadata"]["evidence_type_override"] == "table_text_body_sentence_to_paragraph"


def test_known_false_table_text_patterns_are_demoted() -> None:
    pack = build_evidence_pack(make_clean_doc([
        clean_block("was mixed with 0.5 mL Tris-HCl buffer (200 mM, pH", "table_text", 1),
        clean_block("incubated at 50 °C for 5 min. Then 0.3 mL of substrate", "table_text", 2),
        clean_block("After identification of the sequence encoding active Hac1p we evaluated the effect.", "table_text", 3),
        clean_block("48% byproduct 2’-FL. We assumed that the differences in 2’-FL production between", "table_text", 4),
        clean_block("gDW-1, NGAM=1 mmol ATP gDW-1h-1) for growth in carbon-limited chemostats", "table_text", 5),
    ]))

    assert [unit["type"] for unit in pack["evidence_units"]] == ["paragraph"] * 5
    assert all(unit["metadata"]["source_clean_block_type"] == "table_text" for unit in pack["evidence_units"])


def test_protocol_recipe_table_text_patterns_are_demoted() -> None:
    blocks = [
        clean_block("gradient from 15 to 80% acetonitrile in 0.1% formic acid", "table_text", 1),
        clean_block("umn, 100 Å, 1.8 μm, 300 μm × 150 mm, Waters) using", "table_text", 2),
        clean_block("from 3.5% B (B: 80% ACN, 20% A) to 40% B in 30 min", "table_text", 3),
        clean_block("tryptone, 1 g yeast extract, 34 g NaCl, and 0.1 g FePO4,", "table_text", 4),
        clean_block("(NH4)2SO4, 1 mM MgSO4, 3.9 μM FeSO4, and 1 g (g/l)", "table_text", 5),
    ]
    pack = build_evidence_pack(make_clean_doc(blocks))

    assert [unit["type"] for unit in pack["evidence_units"]] == ["paragraph"] * len(blocks)
    assert all(unit["metadata"]["source_clean_block_type"] == "table_text" for unit in pack["evidence_units"])


def test_phase3a_hotfix4_body_sentences_are_demoted() -> None:
    """Phase3a-hotfix4: body sentences classified as table_text must be demoted to paragraph."""
    blocks = [
        clean_block("confined 46 bacterial species, 42 of which colonize mammals.", "table_text", 1),
        clean_block("strains were grown in lysogeny broth (LB) medium at 37 C.", "table_text", 2),
        clean_block("when paired with the Ost1 signal sequence, secretion improved.", "table_text", 3),
        clean_block("that the NADH/NAD+ ratio was lower in the mutant strain.", "table_text", 4),
        clean_block("GAM and NGAM values were estimated from chemostat data.", "table_text", 5),
        clean_block("in titers up to 120 mg/L, the engineered strain outperformed.", "table_text", 6),
        clean_block("IgG and IgGpAzF titers were measured by ELISA.", "table_text", 7),
        clean_block("by the addition of 1 mM IPTG, expression was induced.", "table_text", 8),
        clean_block("analysis of 83 NanA sequences revealed a distinct clade.", "table_text", 9),
    ]
    pack = build_evidence_pack(make_clean_doc(blocks))

    assert [unit["type"] for unit in pack["evidence_units"]] == ["paragraph"] * len(blocks)
    assert all(unit["metadata"]["source_clean_block_type"] == "table_text" for unit in pack["evidence_units"])


def test_phase3a_hotfix4_real_table_text_still_preserved() -> None:
    """Phase3a-hotfix4: real table text must remain as table_text."""
    pack = build_evidence_pack(make_clean_doc([
        clean_block("Orf_name Length (bp) Encoding protein", "table_text", 1),
        clean_block("gm_orf2729 369 Cytochrome c551", "table_text", 2),
        clean_block("gm_orf521 408 Cytochrome c551", "table_text", 3),
        clean_block("Primer name Sequence OD600 1.0 2.0 3.0", "table_text", 4),
    ]))

    assert [unit["type"] for unit in pack["evidence_units"]] == ["table_text"] * 4
    pack = build_evidence_pack(make_clean_doc([
        clean_block("Useful body evidence.", "paragraph", 1),
        clean_block("S1096-7176(25)00185-5", "paragraph", 2),
        clean_block("This is a PDF of an article that has undergone enhancements after acceptance", "paragraph", 3),
        clean_block("Page 2 of 14", "paragraph", 4),
        clean_block("OPEN ACCESS", "paragraph", 5),
        clean_block("表达 Fam20C 失败,是", "paragraph", 6),
    ]))

    text = "\n".join(unit["text"] for unit in pack["evidence_units"])
    assert text == "Useful body evidence."
    assert pack["excluded_block_counts"]["contamination:cover_metadata"] == 2
    assert pack["excluded_block_counts"]["contamination:running_header_footer"] == 2
    assert pack["excluded_block_counts"]["contamination:annotation_noise"] == 1


def test_evidence_pack_excludes_cover_credit_contamination() -> None:
    pack = build_evidence_pack(make_clean_doc([
        clean_block("Useful evidence remains.", "paragraph", 1),
        clean_block("Metabolic Engineering", "paragraph", 2),
        clean_block("23 November 2023", "paragraph", 3),
        clean_block("in its final form, but we are providing this version to give early visibility of the article. Please note that,", "paragraph", 4),
        clean_block("Shun Endo: Investigation (lead, equal). Sayaka Kamai: Investigation (lead, equal).", "paragraph", 5),
        clean_block("Formal analysis; Writing - original draft. Kento Koketsu: Supervision; Writing - review & editing.", "paragraph", 6),
    ]))

    text = "\n".join(unit["text"] for unit in pack["evidence_units"])
    assert "Useful evidence remains" in text
    assert "Metabolic Engineering" not in text
    assert "23 November 2023" not in text
    assert "early visibility" not in text
    assert "Investigation (lead" not in text
    assert pack["excluded_block_counts"]["contamination:cover_metadata"] == 5


def test_real_table_text_is_not_demoted() -> None:
    pack = build_evidence_pack(make_clean_doc([
        clean_block("Primer Sequence OD600 0.5 1.0 1.5", "table_text", 1),
    ]))

    assert pack["evidence_units"][0]["type"] == "table_text"


def test_evidence_pack_can_be_chunked_without_contamination() -> None:
    pack = build_evidence_pack(make_clean_doc([
        clean_block("Introduction", "section_heading", 1),
        clean_block("Useful fermentation evidence.", "paragraph", 2),
        clean_block("Journal Pre-proof", "metadata", 3),
        clean_block("", "image", 6, metadata_extra={"xref": 42, "image_width": 10, "image_height": 10}),
        clean_block("Table S1: Primers", "table_caption", 4),
        clean_block("Primer Sequence Tm", "table_text", 5),
    ]))
    pages = evidence_pack_to_pages(pack)
    doc = {
        "doc_id": pack["doc_id"],
        "source_file": pack["source_file"],
        "raw_text": "\n".join(page["text"] for page in pages),
        "has_blocks": True,
        "parser_stage": pack["parser_stage"],
        "excluded_block_counts": pack["excluded_block_counts"],
        "pages": pages,
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
    text = "\n".join(chunk.text for chunk in chunks)
    assert "Useful fermentation evidence" in text
    assert "Journal Pre-proof" not in text
    assert "image" not in text.lower()
    assert "[TABLE CAPTION] Table S1: Primers" in text
    assert "[TABLE TEXT] Primer Sequence Tm" in text
    assert chunks[0].excluded_block_counts["metadata"] == 1
    assert chunks[0].excluded_block_counts["image"] == 1
