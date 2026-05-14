from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.synbio_rag.ingestion.cleaning_rules import (
    classify_noise_rule,
    is_false_heading_candidate,
    match_metadata_noise,
    match_reference_noise,
    match_running_header_footer,
)


PROTECTED_EXAMPLES_PATH = Path("results/phase3_cleaning_guardrails/protected_evidence_examples.json")


def _load_protected_examples() -> list[dict]:
    data = json.loads(PROTECTED_EXAMPLES_PATH.read_text(encoding="utf-8"))
    return data["examples"]


@pytest.mark.parametrize(
    "caption",
    [
        "Fig. 1. Engineered metabolic pathways for the production of 3'-sialyllactose.",
        "Figure 5. Subcellular localization of endogenous FAM20A in 17IIA 11 and LS8 cells.",
    ],
)
def test_normal_figure_caption_is_not_noise_or_false_heading(caption: str) -> None:
    assert classify_noise_rule(caption) == (False, "")
    assert is_false_heading_candidate(caption) == (False, "")


@pytest.mark.parametrize(
    "caption",
    [
        "Table 1. Strains and plasmids used in this study.",
        "Table S2. Primers used for ADH6, ADH7, and ADH900 genes.",
    ],
)
def test_normal_table_caption_is_not_noise_or_false_heading(caption: str) -> None:
    assert classify_noise_rule(caption) == (False, "")
    assert is_false_heading_candidate(caption) == (False, "")


def test_markdown_figure_table_headings_are_false_heading_candidates_only_with_heading_marker() -> None:
    assert is_false_heading_candidate("## Figure 1. Workflow")[0]
    assert is_false_heading_candidate("### Table S2. Primers")[0]
    assert is_false_heading_candidate("Figure 1. Workflow") == (False, "")
    assert is_false_heading_candidate("Table S2. Primers") == (False, "")


@pytest.mark.parametrize(
    "text",
    [
        "The NFS-60 cell line was grown in medium supplemented with 8% horse serum and 2% fetal bovine serum.",
        "GDP-L-fucose concentration of 38.9±0.05 mg l−1 was obtained with productivity of 1.8 mg l−1 h−1.",
        "Fed-batch fermentation started with 100 mL culture at 30 °C for 16 h until OD600 reached 2.0.",
    ],
)
def test_numeric_unit_evidence_is_not_metadata_or_running_header(text: str) -> None:
    assert match_metadata_noise(text) == (False, "")
    assert match_running_header_footer(text) == (False, "")
    assert classify_noise_rule(text) == (False, "")


@pytest.mark.parametrize(
    "text",
    [
        "The upstream primer was 5′CACTGGCGATTGATATCGGCGGTACTAAACTTGCCGCCGTGTAGGCTGGAGCTGCTTC.",
        "An E. coli strain BL21(DE3) was used as a host, and pETGW harbors the dicistronic gene cluster.",
    ],
)
def test_primer_sequence_and_strain_vector_lines_are_not_metadata_noise(text: str) -> None:
    assert match_metadata_noise(text) == (False, "")
    assert classify_noise_rule(text) == (False, "")


def test_references_heading_is_detected() -> None:
    assert match_reference_noise("References") == (True, "reference_section")
    assert classify_noise_rule("2. Literature Cited") == (True, "reference_section")


@pytest.mark.parametrize("text", ["E-mail address: author@example.org", "Email address: author@example.org"])
def test_email_address_variants_are_metadata(text: str) -> None:
    assert match_metadata_noise(text) == (True, "metadata_correspondence")


@pytest.mark.parametrize("heading", ["Results", "Materials and Methods", "1. Introduction"])
def test_normal_section_heading_is_kept(heading: str) -> None:
    assert classify_noise_rule(heading) == (False, "")
    assert is_false_heading_candidate(heading) == (False, "")


@pytest.mark.parametrize(
    ("text", "rule_id"),
    [
        ("© 2024 The Authors", "metadata_copyright"),
        ("All rights reserved.", "metadata_copyright"),
        ("Open Access", "metadata_open_access"),
    ],
)
def test_copyright_and_open_access_metadata_are_removed(text: str, rule_id: str) -> None:
    assert match_metadata_noise(text) == (True, rule_id)


def test_protected_keep_examples_do_not_match_cleaning_noise_rules() -> None:
    failures = []
    for example in _load_protected_examples():
        if example["expected_cleaning_decision"] != "keep":
            continue
        matched, rule_id = classify_noise_rule(example["text"])
        false_heading, false_heading_rule = is_false_heading_candidate(example["text"])
        if matched or false_heading:
            failures.append({
                "example_id": example["example_id"],
                "rule_id": rule_id or false_heading_rule,
                "text": example["text"],
            })
    assert failures == []


def test_protected_remove_examples_still_match_cleaning_rules() -> None:
    remove_examples = [
        example
        for example in _load_protected_examples()
        if example["expected_cleaning_decision"] == "remove"
    ]
    assert remove_examples
    assert all(classify_noise_rule(example["text"])[0] for example in remove_examples)
