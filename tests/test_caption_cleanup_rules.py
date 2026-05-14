from __future__ import annotations

from src.synbio_rag.ingestion.caption_cleanup import (
    SAFE_LABEL,
    SignoffDecision,
    apply_cleanup_to_doc,
)


def decision(
    *,
    doc_id: str = "doc_test",
    block_id: str = "p1_b0002",
    block_type: str = "table_caption",
    caption_text: str = "Table 1. E.",
    candidate_rule: str = "broken_organism_or_abbreviation_prefix",
    label: str = SAFE_LABEL,
    confidence: str = "high",
) -> SignoffDecision:
    return SignoffDecision(
        doc_id=doc_id,
        block_id=block_id,
        block_type=block_type,
        caption_text=caption_text,
        candidate_rule=candidate_rule,
        confidence=confidence,
        label=label,
        rationale="test",
        recommended_cleanup_action="demote_fragment_caption_preserve_text_with_cleanup_metadata",
        risk_if_wrong="test risk",
    )


def block(block_id: str, block_type: str, text: str) -> dict:
    return {
        "block_id": block_id,
        "type": block_type,
        "text": text,
        "page": 1,
        "section_path": ["Methods"],
    }


def doc(blocks: list[dict], *, doc_id: str = "doc_test") -> dict:
    return {
        "doc_id": doc_id,
        "source_file": f"{doc_id}.pdf",
        "parser_stage": "parsed_clean_v1",
        "pages": [{"page": 1, "text": "\n".join(b["text"] for b in blocks), "blocks": blocks}],
    }


def cleanup_one(
    blocks: list[dict],
    signoff: SignoffDecision,
    *,
    doc_id: str = "doc_test",
    protected: bool = False,
) -> tuple[dict, list]:
    protected_keys = {(doc_id, signoff.block_id)} if protected else set()
    new_doc, audit, _ = apply_cleanup_to_doc(
        doc(blocks, doc_id=doc_id),
        {(doc_id, signoff.block_id): signoff},
        protected_keys,
    )
    return new_doc, audit


def target_block(new_doc: dict, block_id: str = "p1_b0002") -> dict:
    for page in new_doc["pages"]:
        for item in page["blocks"]:
            if item["block_id"] == block_id:
                return item
    raise AssertionError(f"missing block {block_id}")


def test_article_plus_single_letter_fragment_is_demoted() -> None:
    blocks = [
        block("p1_b0001", "paragraph", "The strains used are listed in"),
        block("p1_b0002", "table_caption", "Table S1. The E."),
        block("p1_b0003", "paragraph", "coli DH5α strain was used for cloning."),
    ]
    signoff = decision(caption_text="Table S1. The E.", candidate_rule="article_plus_single_letter_fragment")

    new_doc, audit = cleanup_one(blocks, signoff)

    cleaned = target_block(new_doc)
    assert cleaned["type"] == "paragraph"
    assert cleaned["text"] == "Table S1. The E."
    assert cleaned["metadata"]["original_block_type"] == "table_caption"
    assert cleaned["metadata"]["caption_cleanup_rule_id"] == "phase5d3_article_plus_single_letter_fragment"
    assert cleaned["metadata"]["caption_cleanup_reason"]
    assert audit[0].cleanup_action == "demote_to_paragraph"


def test_broken_organism_prefix_with_strong_nearby_evidence_is_demoted() -> None:
    blocks = [
        block("p1_b0001", "paragraph", "All strains and plasmids used in this study are listed in"),
        block("p1_b0002", "table_caption", "Table 1. E."),
        block("p1_b0003", "paragraph", "coli BL21(DE3) was used for protein expression."),
    ]
    signoff = decision()

    new_doc, _ = cleanup_one(blocks, signoff)

    assert target_block(new_doc)["type"] == "paragraph"


def test_number_only_caption_broad_case_is_not_demoted() -> None:
    blocks = [
        block("p1_b0001", "paragraph", "The result is shown in Figure 3."),
        block("p1_b0002", "figure_caption", "Figure 3."),
        block("p1_b0003", "paragraph", "Growth profiles of engineered strains."),
    ]
    signoff = decision(
        block_type="figure_caption",
        caption_text="Figure 3.",
        candidate_rule="number_only_caption",
    )

    new_doc, audit = cleanup_one(blocks, signoff)

    assert target_block(new_doc)["type"] == "figure_caption"
    assert audit[0].cleanup_action == "skip_number_only_guard"


def test_very_short_no_semantic_anchor_broad_case_is_not_demoted() -> None:
    blocks = [
        block("p1_b0001", "paragraph", "Some context."),
        block("p1_b0002", "figure_caption", "Fig. 2. 2."),
        block("p1_b0003", "paragraph", "More context."),
    ]
    signoff = decision(
        block_type="figure_caption",
        caption_text="Fig. 2. 2.",
        candidate_rule="very_short_no_semantic_anchor",
    )

    new_doc, audit = cleanup_one(blocks, signoff)

    assert target_block(new_doc)["type"] == "figure_caption"
    assert audit[0].cleanup_action == "skip_rule_guard"


def test_page_header_footer_fragment_is_demoted() -> None:
    blocks = [
        block("p1_b0001", "paragraph", "Some context."),
        block("p1_b0002", "figure_caption", "FIGURE 3 5 of 12"),
        block("p1_b0003", "paragraph", "More context."),
    ]
    signoff = decision(
        block_type="figure_caption",
        caption_text="FIGURE 3 5 of 12",
        candidate_rule="very_short_no_semantic_anchor",
        confidence="medium",
    )

    new_doc, _ = cleanup_one(blocks, signoff)

    assert target_block(new_doc)["type"] == "paragraph"


def test_figure_continued_caption_is_not_demoted() -> None:
    blocks = [
        block("p1_b0001", "paragraph", "Some context."),
        block("p1_b0002", "figure_caption", "Figure 3. Cont."),
        block("p1_b0003", "paragraph", "More context."),
    ]
    signoff = decision(
        block_type="figure_caption",
        caption_text="Figure 3. Cont.",
        candidate_rule="very_short_no_semantic_anchor",
    )

    new_doc, audit = cleanup_one(blocks, signoff)

    assert target_block(new_doc)["type"] == "figure_caption"
    assert audit[0].cleanup_action == "skip_continued_guard"


def test_semantic_anchor_short_caption_is_not_demoted() -> None:
    blocks = [
        block("p1_b0001", "paragraph", "Some context."),
        block("p1_b0002", "figure_caption", "Fig. 1. Workflow."),
        block("p1_b0003", "paragraph", "More context."),
    ]
    signoff = decision(
        block_type="figure_caption",
        caption_text="Fig. 1. Workflow.",
        candidate_rule="very_short_no_semantic_anchor",
    )

    new_doc, audit = cleanup_one(blocks, signoff)

    assert target_block(new_doc)["type"] == "figure_caption"
    assert audit[0].cleanup_action == "skip_descriptive_caption"


def test_protected_caption_is_not_demoted() -> None:
    blocks = [
        block("p1_b0001", "paragraph", "All strains are listed in"),
        block("p1_b0002", "table_caption", "Table 1. E."),
        block("p1_b0003", "paragraph", "coli strain details follow."),
    ]
    signoff = decision()

    new_doc, audit = cleanup_one(blocks, signoff, protected=True)

    assert target_block(new_doc)["type"] == "table_caption"
    assert audit[0].cleanup_action == "skip_protected"


def test_doc_0367_like_descriptive_caption_is_not_demoted() -> None:
    blocks = [
        block("p1_b0001", "paragraph", "Some context."),
        block(
            "p1_b0002",
            "figure_caption",
            "Figure 5. Comparison of Opto-T7RNAPs to paT7P-148.",
        ),
        block("p1_b0003", "paragraph", "More context."),
    ]
    signoff = decision(
        doc_id="doc_0367",
        block_type="figure_caption",
        caption_text="Figure 5. Comparison of Opto-T7RNAPs to paT7P-148.",
        candidate_rule="very_short_no_semantic_anchor",
    )

    new_doc, audit = cleanup_one(blocks, signoff, doc_id="doc_0367")

    assert target_block(new_doc)["type"] == "figure_caption"
    assert audit[0].cleanup_action == "skip_descriptive_caption"
