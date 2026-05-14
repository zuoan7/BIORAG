from __future__ import annotations

import pytest

from src.synbio_rag.ingestion.cleaning_rules import (
    CleaningContext,
    classify_noise_rule,
    classify_noise_rule_with_context,
    is_false_heading_candidate,
    is_false_heading_with_context,
    looks_like_affiliation_or_address,
    looks_like_back_matter_metadata,
    looks_like_body_fragment_heading,
    looks_like_reference_entry_with_context,
    match_journal_preproof_noise,
    match_metadata_noise,
    match_reference_noise,
    match_running_header_footer,
    normalize_cleaning_text,
)


@pytest.mark.parametrize(
    ("text", "rule_id"),
    [
        ("Journal Pre-proof", "journal_preproof_exact"),
        ("This is a PDF file of an article that has undergone enhancements", "journal_preproof_disclaimer"),
        ("PII: S1096-7176(25)00185-5", "journal_preproof_metadata"),
    ],
)
def test_journal_preproof_noise_rules(text: str, rule_id: str) -> None:
    assert match_journal_preproof_noise(text) == (True, rule_id)


@pytest.mark.parametrize(
    ("text", "rule_id"),
    [
        ("DOI: 10.1016/example", "metadata_doi"),
        ("https://doi.org/10.1016/example", "metadata_doi"),
        ("*Corresponding author: author@example.org", "metadata_correspondence"),
        ("E-mail address: author@example.org", "metadata_correspondence"),
        ("© 2024 The Authors", "metadata_copyright"),
        ("All rights reserved.", "metadata_copyright"),
        ("Open Access", "metadata_open_access"),
    ],
)
def test_metadata_noise_rules(text: str, rule_id: str) -> None:
    assert match_metadata_noise(text) == (True, rule_id)


@pytest.mark.parametrize(
    "text",
    [
        "Page 2 of 14",
        "OPEN ACCESS",
        "Biotechnology and Bioengineering, Vol. 110, No. 3",
        "Barrero et al. Microb Cell Fact (2018) 17:161",
    ],
)
def test_running_header_footer_rules(text: str) -> None:
    assert match_running_header_footer(text) == (True, "running_header_footer")


@pytest.mark.parametrize(
    "text",
    [
        "References",
        "Bibliography",
        "Literature Cited",
        "2. Works Cited",
    ],
)
def test_reference_heading_rules(text: str) -> None:
    assert match_reference_noise(text) == (True, "reference_section")


@pytest.mark.parametrize(
    ("text", "rule_id"),
    [
        ("## 16S rRNA gene sequencing", "false_heading_author_line"),
        ("## 27F", "false_heading_author_line"),
        ("## Figure 1. Workflow", "false_heading_table_or_figure"),
        ("### Table S2. Primers", "false_heading_table_or_figure"),
        ("## Correspondence and requests for materials", "metadata_correspondence"),
    ],
)
def test_false_heading_rules(text: str, rule_id: str) -> None:
    assert is_false_heading_candidate(text) == (True, rule_id)


@pytest.mark.parametrize(
    "text",
    [
        "The engineered strain produced 3.2 g/L lactate after 48 h in fed-batch fermentation.",
        "Results",
        "1. Introduction",
        "Materials and Methods",
        "Fig. 1. Overview of the fermentation workflow.",
        "Table 1. Strains and plasmids used in this study.",
    ],
)
def test_normal_evidence_and_headings_are_not_noise(text: str) -> None:
    assert classify_noise_rule(text) == (False, "")
    assert is_false_heading_candidate(text) == (False, "")


def test_normalize_cleaning_text_handles_pdf_unicode_noise() -> None:
    assert normalize_cleaning_text("Fig.\u00a01 \u2013 workflow\u200b") == "Fig. 1 - workflow"


def test_front_matter_affiliation_address_requires_context() -> None:
    text = "1 Department of Food Engineering, Akdeniz University, Antalya, Turkey"

    assert looks_like_affiliation_or_address(text, CleaningContext()) == (False, "")
    assert looks_like_affiliation_or_address(
        text,
        CleaningContext(page=1, in_front_matter=True, y0=120, y1=140, column="L"),
    ) == (True, "context_affiliation_address")


@pytest.mark.parametrize("heading", ["Introduction", "Results", "Methods", "Materials and Methods"])
def test_context_rules_keep_normal_section_headings(heading: str) -> None:
    context = CleaningContext(block_type="section_heading", page=4)

    assert is_false_heading_with_context(heading, context) == (False, "")
    assert classify_noise_rule_with_context(heading, context) == (False, "")


@pytest.mark.parametrize(
    "heading",
    [
        "7. Rabinowitz, M., and Lipmann, F. (1960) Reversible phosphate",
        "5A,B). The pykA-knockout strain produced 9.9 g/L",
        "1950 V; and mass range, 20-2000 m/z.",
    ],
)
def test_context_body_fragment_heading_is_false_heading(heading: str) -> None:
    context = CleaningContext(block_type="section_heading", page=5)

    assert looks_like_body_fragment_heading(heading, context) == (True, "context_body_fragment_heading")
    expected_rule = (
        "reference_entry"
        if heading.startswith("7. Rabinowitz")
        else "context_body_fragment_heading"
    )
    assert is_false_heading_with_context(heading, context) == (True, expected_rule)


def test_author_contribution_metadata_requires_back_matter_context() -> None:
    text = "Formal analysis; Writing - original draft; Funding acquisition"

    assert looks_like_back_matter_metadata(text, CleaningContext()) == (False, "")
    assert looks_like_back_matter_metadata(
        text,
        CleaningContext(
            block_type="paragraph",
            page=12,
            section_path=["Discussion", "CRediT authorship contribution statement"],
        ),
    ) == (True, "context_author_contribution")


def test_credit_authorship_contribution_still_matches_author_contribution() -> None:
    text = "Jane Doe: Formal analysis, Methodology, Writing - original draft."
    context = CleaningContext(
        block_type="paragraph",
        page=12,
        section_path=["CRediT authorship contribution statement"],
    )

    assert classify_noise_rule_with_context(text, context) == (True, "context_author_contribution")


@pytest.mark.parametrize(
    "text",
    [
        "The ecoinvent database version 3 (part I): overview and methodology. Int. J. Life Cycle Assess. 21, 1218-1230.",
        "Chaudhry, M.T., Huang, Y., Shen, X.H., Poetsch, A., Jiang, C.Y., Liu, S.J., 2007. Genome-wide investigation of aromatic acid transporters in Corynebacterium glutamicum.",
        "[89] Singh G, Singh S, Kaur K, Arya SK, Sharma P. Thermo and halo tolerant laccase from Bacillus sp. SS4. J Gen Appl Microbiol 2019;65:26-33.",
    ],
)
def test_reference_entries_do_not_match_author_contribution(text: str) -> None:
    context = CleaningContext(block_type="references", page=12, in_references=True)

    assert looks_like_back_matter_metadata(text, context) == (False, "")
    assert classify_noise_rule_with_context(text, context) == (True, "reference_entry")


def test_references_section_path_prioritizes_reference_rule() -> None:
    text = "Formal analysis of prior work is discussed in Smith et al. J. Biotechnol 2020;12:1-9."
    context = CleaningContext(block_type="paragraph", page=10, section_path=["References"])

    assert looks_like_reference_entry_with_context(text, context) == (True, "reference_entry")
    assert classify_noise_rule_with_context(text, context) == (True, "reference_entry")


def test_normal_body_text_is_not_reference_entry() -> None:
    text = "The engineered strain produced 3.2 g/L lactate after 48 h in fed-batch fermentation."
    context = CleaningContext(block_type="paragraph", page=4, section_path=["Results"])

    assert looks_like_reference_entry_with_context(text, context) == (False, "")
    assert classify_noise_rule_with_context(text, context) == (False, "")


@pytest.mark.parametrize(
    "caption",
    [
        "Fig. 2. Engineered pathway overview.",
        "Table S2. Primers used in this study.",
    ],
)
def test_context_false_heading_keeps_figure_table_captions(caption: str) -> None:
    context = CleaningContext(block_type="section_heading", page=5)

    assert is_false_heading_with_context(caption, context) == (False, "")
    assert classify_noise_rule_with_context(caption, context) == (False, "")


def test_context_rules_are_conservative_without_context() -> None:
    assert classify_noise_rule_with_context(
        "Department activity increased after pathway optimization.",
        CleaningContext(),
    ) == (False, "")
    assert looks_like_body_fragment_heading(
        "1950 V; and mass range, 20-2000 m/z.",
        CleaningContext(block_type="paragraph"),
    ) == (False, "")
