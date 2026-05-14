from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


CONTROL_CHARS_RE = re.compile(r"[\u0000-\u0008\u000b-\u001f\u007f-\u009f]")
ZERO_WIDTH_CHARS_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
UNICODE_SPACE_CHARS_RE = re.compile(r"[\u00a0\u202f\u2009\u2007]")
DASH_CHARS_RE = re.compile(r"[\u2013\u2014\u2212]")

JOURNAL_PREPROOF_EXACT_RE = re.compile(r"^journal pre-proofs?$", re.I)
JOURNAL_PREPROOF_METADATA_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^PII\s*:", re.I), "journal_preproof_metadata"),
    (re.compile(r"^DOI\s*:", re.I), "metadata_doi"),
    (re.compile(r"^Reference\s*:", re.I), "journal_preproof_metadata"),
    (re.compile(r"^To appear in\s*:", re.I), "journal_preproof_metadata"),
    (re.compile(r"^(?:Received|Revised|Accepted) Date\s*:", re.I), "journal_preproof_metadata"),
]
JOURNAL_PREPROOF_DISCLAIMER_PATTERNS: list[re.Pattern] = [
    re.compile(r"^This is a PDF file of an article", re.I),
    re.compile(r"^This is a PDF of an article", re.I),
    re.compile(r"^Please cite this article as:", re.I),
    re.compile(r"^This manuscript has been accepted", re.I),
    re.compile(r"^The manuscript will undergo copyediting", re.I),
    re.compile(r"\bthis version will undergo additional copyediting\b", re.I),
    re.compile(r"\bduring the production process, errors may be discovered\b", re.I),
    re.compile(r"\blegal disclaimers that apply to the journal pertain\b", re.I),
    re.compile(r"\bin its final form, but we are providing this version\b", re.I),
    re.compile(r"\bthis early version to give early visibility of the article\b", re.I),
    re.compile(r"\bplease note that\b.*\bearly visibility of the article\b", re.I),
    re.compile(r"\bplease also note that\b.*\bduring the production process\b", re.I),
]

METADATA_NOISE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(?:https?://doi\.org/|doi\s*:|doi\b)", re.I), "metadata_doi"),
    (re.compile(r"^https?://", re.I), "metadata_url"),
    (re.compile(r"^www\.", re.I), "metadata_url"),
    (re.compile(r"^\*?\s*Corresponding author\b", re.I), "metadata_correspondence"),
    (re.compile(r"^Correspondence\s*:|^Correspondence and requests", re.I), "metadata_correspondence"),
    (re.compile(r"^To whom correspondence should be addressed\b", re.I), "metadata_correspondence"),
    (re.compile(r"^Correspondence may also be addressed\b", re.I), "metadata_correspondence"),
    (re.compile(r"^Should be addressed to\b", re.I), "metadata_correspondence"),
    (re.compile(r"^E-?mail(?: address(?:es)?)?\s*:", re.I), "metadata_correspondence"),
    (re.compile(r"^(?:Received|Accepted|Revised|Published)(?:\s+in revised form)?\b", re.I), "metadata_dates"),
    (re.compile(r"^Available online\b", re.I), "metadata_dates"),
    (re.compile(r"^Contents lists available at\b", re.I), "metadata_journal_sidebar"),
    (re.compile(r"^journal homepage\s*:", re.I), "metadata_journal_sidebar"),
    (re.compile(r"^(?:©\s*)?\d{4}-\d{4}/©\s*\d{4}", re.I), "metadata_copyright"),
    (re.compile(r"^©\s*\d{4}\b", re.I), "metadata_copyright"),
    (re.compile(r"^All rights reserved\.?$", re.I), "metadata_copyright"),
    (re.compile(r"^Please cite this article\b", re.I), "metadata_citation_notice"),
    (re.compile(r"^Downloaded from\b|^Downloaded by\b", re.I), "metadata_access_banner"),
    (re.compile(r"^Supplementary (?:material|data|information)\b", re.I), "metadata_supplementary"),
    (re.compile(r"^Subject areas\b", re.I), "metadata_subject_area"),
    (re.compile(r"^Open access$", re.I), "metadata_open_access"),
]

RUNNING_HEADER_FOOTER_PATTERNS: list[re.Pattern] = [
    re.compile(r"^page\s+\d+\s+of\s+\d+$", re.I),
    re.compile(r"^open access$", re.I),
    re.compile(r"^vol\.\s*\d+,\s*no\.\s*\d+(?:,\s*\d{4})?$", re.I),
    re.compile(r"^j\.\s+biochem\.", re.I),
    re.compile(r"^biotechnology\s+and\s+bioengineering,\s+vol\.\s*110,\s+no\.\s*3", re.I),
    re.compile(r"^barrero\s+et\s+al\.\s+microb\s+cell\s+fact\s+\(\d{4}\)\s+\d+:\d+$", re.I),
    re.compile(r"^zhu\s+et\s+al\.\s+biotechnol\s+biofuels\s+\(\d{4}\)\s+\d+:\d+$", re.I),
]

REFERENCE_SECTION_RE = re.compile(
    r"^(?:\d+\.?\s+)?(references|bibliography|literature cited|works cited)$",
    re.I,
)

CONTAMINATION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"journal pre-proof|this is a pdf file of an article", re.I), "journal_preproof_disclaimer"),
    (
        re.compile(
            r"\b(?:\*?\s*correspondence\s*:|to whom correspondence|correspondence may also)",
            re.I,
        ),
        "metadata_correspondence",
    ),
    (
        re.compile(
            r"S1096-7176|\bYMBEN\b|Accepted Manuscript|Version of Record|"
            r"this is a pdf of an article|"
            r"this version will undergo additional copyediting|"
            r"of a cover page and metadata|"
            r"during the production process, errors may be discovered|"
            r"in its final form, but we are providing this version|"
            r"this early version to give early visibility of the article|"
            r"please note that.*early visibility of the article|"
            r"please also note that.*during the production process|"
            r"errors may be discovered which could affect the content|"
            r"disclaimers that apply to the journal pertain|"
            r"all legal disclaimers that apply to the journal|"
            r"^Metabolic Engineering$|^\d{1,2}\s+November\s+2023$|"
            r"\b(?:Investigation|Formal analysis|Conceptualization|Supervision|Writing\s*-\s*original draft|"
            r"Writing\s*-\s*review\s*&\s*editing|Methodology|Validation|Visualization|Funding acquisition)\b",
            re.I,
        ),
        "contamination_cover_metadata",
    ),
    (
        re.compile(
            r"表达\s*Fam20C|是否有尝试|共表达\s*\?{2,}|[\u4e00-\u9fff]{2,}.{0,20}\?{2,}"
        ),
        "contamination_annotation_noise",
    ),
]

FALSE_HEADING_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^#{1,3}\s+\d+S\s+rRNA", re.I), "false_heading_author_line"),
    (re.compile(r"^#{1,3}\s+\d+F\b"), "false_heading_author_line"),
    (re.compile(r"^#{1,3}\s+13CH", re.I), "false_heading_author_line"),
    (re.compile(r"^#{1,3}\s+1F-β?-?fructofuranosyl", re.I), "false_heading_author_line"),
    (re.compile(r"^#{1,3}\s+\d+F-"), "false_heading_author_line"),
    (re.compile(r"^#{1,3}\s+\d+R\b"), "false_heading_author_line"),
    (re.compile(r"^#{1,3}\s+\d+C-"), "false_heading_author_line"),
    (re.compile(r"^#{1,3}\s*(?:FIGURE|TABLE)\s*\d+", re.I), "false_heading_table_or_figure"),
    (re.compile(r"^#{1,3}\s*(?:Fig\.?|Figure|Table)\s+S?\d+", re.I), "false_heading_table_or_figure"),
]

CONTEXT_CORRESPONDENCE_RE = re.compile(
    r"^\s*(?:[*†‡§y]\s*)?(?:"
    r"correspondence\s*:|"
    r"corresponding\s+authors?\b|"
    r"to\s+whom\s+correspondence\s+should\s+be\s+addressed\b|"
    r"correspondence\s+may\s+also\s+be\s+addressed\b|"
    r"e-?mail(?:\s+addresses?)?\s*:|"
    r"current\s+address\s*:|present\s+address\s*:|co-first\s+authors?\b"
    r")",
    re.I,
)
CONTEXT_NUMBERED_AFFILIATION_RE = re.compile(
    r"^\s*\d{1,2}\s+(?:"
    r"department|university|institute|laborator(?:y|ies)|division|center|centre|"
    r"faculty|school|college|hospital"
    r")\b",
    re.I,
)
CONTEXT_INSTITUTION_WORD_RE = re.compile(
    r"\b(?:"
    r"department|university|institute|laborator(?:y|ies)|division|center|centre|"
    r"faculty|school|college|hospital|biosustainability|biorefining|technology"
    r")\b",
    re.I,
)
CONTEXT_ADDRESS_HINT_RE = re.compile(
    r"\b(?:"
    r"china|usa|u\.s\.a\.|denmark|sweden|france|netherlands|germany|japan|"
    r"belgium|italy|spain|turkey|canada|australia|kingdom|states|"
    r"\d{4,6}|@"
    r")\b",
    re.I,
)
CONTEXT_AUTHOR_CONTRIBUTION_RE = re.compile(
    r"\b(?:Investigation|Formal analysis|Conceptualization|Supervision|"
    r"Writing\s*-\s*original draft|Writing\s*-\s*review\s*&\s*editing|"
    r"Methodology|Validation|Visualization|Funding acquisition)\b",
    re.I,
)
CONTEXT_BACK_MATTER_HEADING_RE = re.compile(
    r"^(?:#+\s*)?(?:"
    r"acknowledgments?|acknowledgements?|funding|data availability(?: statement)?|"
    r"author contributions?|contributors|additional information|"
    r"supplementary (?:information|data|materials?)|supporting information|"
    r"conflicts? of interest|competing (?:financial )?interests?|"
    r"ethics (?:statement|approval)|license|publisher'?s? note|author information"
    r")\s*:?\s*$",
    re.I,
)
REFERENCE_CONTEXT_RE = re.compile(r"\b(?:references|bibliography|literature cited|works cited)\b", re.I)
REFERENCE_NUMBERED_ENTRY_RE = re.compile(r"^\s*(?:\[\d+\]|\d{1,3}\.)\s+\S+", re.I)
REFERENCE_AUTHOR_YEAR_RE = re.compile(
    r"^[A-Z][A-Za-z'’`-]+,\s+(?:[A-Z]\.|[A-Z][A-Za-z'’`-]+).{0,220}\b(?:19|20)\d{2}\b",
    re.I,
)
REFERENCE_JOURNAL_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"J\.|Journal|Proc\.|Nat\.|Science|Cell|Microbiol|Biotechnol|Biofuels|"
    r"Bioresour|Chem|Eng|Energy|Environ|Front|Appl|Int\.|ACS|PLoS|BMC|"
    r"Life Cycle Assess|Sustainable"
    r")\b",
    re.I,
)
REFERENCE_VOLUME_PAGE_RE = re.compile(
    r"\b\d{1,4}\s*(?:\(\d+\))?\s*[,;:]\s*\d{1,5}(?:[–-]\d{1,5})?\b"
)
REFERENCE_DOI_RE = re.compile(r"\b(?:doi|https?://doi\.org/|10\.\d{4,9}/)\b", re.I)
CONTEXT_BODY_FRAGMENT_HEADING_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*\d+\.\s+[A-Z][a-z'-]+,\s+[A-Z]\.", re.I),
    re.compile(r"^\s*\d{4,6}\s+\w+,\s+\w+", re.I),
    re.compile(r"^\s*\d+\s*V\s*;", re.I),
    re.compile(r"^\s*\d{8,}[)）]", re.I),
    re.compile(r"^\s*\d+[A-F](?:,[A-F])*[)）]\.?\s+[a-z]", re.I),
]


@dataclass
class CleaningContext:
    block_type: str = ""
    page: int | None = None
    y0: float | None = None
    y1: float | None = None
    column: str = ""
    section_path: list[str] | None = None
    previous_text: str = ""
    next_text: str = ""
    previous_block_type: str = ""
    next_block_type: str = ""
    in_front_matter: bool = False
    in_references: bool = False
    recent_table_caption: bool = False


def normalize_cleaning_text(text: str) -> str:
    """Normalize PDF text for shared cleaning rules and return a compact string."""
    text = unicodedata.normalize("NFKC", text or "")
    text = UNICODE_SPACE_CHARS_RE.sub(" ", text)
    text = DASH_CHARS_RE.sub("-", text)
    text = text.replace("\u00ad", "")
    text = ZERO_WIDTH_CHARS_RE.sub("", text)
    text = CONTROL_CHARS_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _headingless(text: str) -> str:
    return normalize_cleaning_text(text).lstrip("#").strip()


def match_journal_preproof_noise(text: str) -> tuple[bool, str]:
    """Return whether text is Journal Pre-proof noise and its stable rule_id."""
    normalized = _headingless(text)
    if JOURNAL_PREPROOF_EXACT_RE.match(normalized):
        return True, "journal_preproof_exact"
    for pattern, rule_id in JOURNAL_PREPROOF_METADATA_PATTERNS:
        if pattern.search(normalized):
            return True, rule_id
    for pattern in JOURNAL_PREPROOF_DISCLAIMER_PATTERNS:
        if pattern.search(normalized):
            return True, "journal_preproof_disclaimer"
    return False, ""


def match_metadata_noise(text: str) -> tuple[bool, str]:
    """Return whether text is common metadata noise and its stable rule_id."""
    normalized = _headingless(text)
    for pattern, rule_id in METADATA_NOISE_PATTERNS:
        if pattern.search(normalized):
            return True, rule_id
    return False, ""


def match_running_header_footer(text: str) -> tuple[bool, str]:
    """Return whether text looks like a running header/footer and its rule_id."""
    normalized = _headingless(text)
    for pattern in RUNNING_HEADER_FOOTER_PATTERNS:
        if pattern.search(normalized):
            return True, "running_header_footer"
    return False, ""


def _is_marginal_or_header_position(context: CleaningContext) -> bool:
    return (
        context.y0 is not None
        and context.y1 is not None
        and (context.y0 < 80 or context.y1 > 700)
    )


def _section_path_text(context: CleaningContext) -> str:
    return " > ".join(context.section_path or [])


def _has_reference_context(context: CleaningContext) -> bool:
    return context.in_references or bool(REFERENCE_CONTEXT_RE.search(_section_path_text(context)))


def _looks_like_reference_entry_text(text: str, require_context: bool = False) -> bool:
    normalized = _headingless(text)
    if not normalized:
        return False
    if match_reference_noise(normalized)[0]:
        return True
    if REFERENCE_NUMBERED_ENTRY_RE.match(normalized):
        return True

    feature_count = 0
    if REFERENCE_AUTHOR_YEAR_RE.search(normalized):
        feature_count += 1
    if REFERENCE_JOURNAL_SIGNAL_RE.search(normalized):
        feature_count += 1
    if REFERENCE_VOLUME_PAGE_RE.search(normalized):
        feature_count += 1
    if REFERENCE_DOI_RE.search(normalized):
        feature_count += 1
    if re.search(r"\bet\s+al\.?\b", normalized, re.I):
        feature_count += 1
    if re.search(r"\[(?:\d{1,3})\]", normalized):
        feature_count += 1

    threshold = 1 if require_context else 2
    return feature_count >= threshold


def looks_like_reference_entry_with_context(text: str, context: CleaningContext) -> tuple[bool, str]:
    """Return whether text should be audited as a references entry/section."""
    in_reference_context = _has_reference_context(context)
    if match_reference_noise(text)[0]:
        return True, "reference_section"
    if in_reference_context and normalize_cleaning_text(text):
        return True, "reference_entry"
    if _looks_like_reference_entry_text(text, require_context=in_reference_context):
        return True, "reference_entry"
    return False, ""


def looks_like_affiliation_or_address(text: str, context: CleaningContext) -> tuple[bool, str]:
    """Return whether front-matter context makes text look like affiliation metadata."""
    normalized = normalize_cleaning_text(text)
    if not normalized:
        return False, ""
    if CONTEXT_CORRESPONDENCE_RE.match(normalized):
        return True, "context_correspondence_metadata"
    if not context.in_front_matter:
        return False, ""
    words = normalized.split()
    short_line = len(words) <= 12 or len(normalized) <= 110
    has_institution = bool(CONTEXT_INSTITUTION_WORD_RE.search(normalized))
    has_address_hint = bool(CONTEXT_ADDRESS_HINT_RE.search(normalized))
    side_column = context.column in {"L", "R", "MARGIN", "UNK"} and _is_marginal_or_header_position(context)
    if CONTEXT_NUMBERED_AFFILIATION_RE.match(normalized):
        return True, "context_affiliation_address"
    if re.match(r"^\s*\d{1,2}\s+", normalized) and has_institution and short_line:
        return True, "context_affiliation_address"
    if has_institution and short_line and (has_address_hint or side_column or normalized.endswith(",")):
        return True, "context_affiliation_address"
    if re.match(r"^\s*(?:full list of author information|co-first authors?)\b", normalized, re.I):
        return True, "context_affiliation_address"
    return False, ""


def looks_like_back_matter_metadata(text: str, context: CleaningContext) -> tuple[bool, str]:
    """Return whether context indicates author/funding/back-matter metadata."""
    normalized = _headingless(text)
    if not normalized:
        return False, ""
    if looks_like_reference_entry_with_context(normalized, context)[0]:
        return False, ""
    in_back_matter = context.in_references or bool(
        re.search(
            r"\b(?:"
            r"acknowledg|funding|author contributions?|authorship contribution|"
            r"credit authorship|competing interests?|data availability"
            r")\b",
            _section_path_text(context),
            re.I,
        )
    )
    if in_back_matter and CONTEXT_AUTHOR_CONTRIBUTION_RE.search(normalized):
        return True, "context_author_contribution"
    if CONTEXT_BACK_MATTER_HEADING_RE.match(normalized) and (
        in_back_matter or context.block_type in {"section_heading", "subsection_heading"}
    ):
        return True, "context_back_matter_metadata"
    return False, ""


def looks_like_body_fragment_heading(text: str, context: CleaningContext) -> tuple[bool, str]:
    """Return whether heading-like text is a body/reference/address fragment."""
    normalized = _headingless(text)
    if not normalized:
        return False, ""
    if context.recent_table_caption or context.block_type not in {"section_heading", "subsection_heading", "title", ""}:
        return False, ""
    if match_reference_noise(normalized)[0] or match_metadata_noise(normalized)[0]:
        return False, ""
    for pattern in CONTEXT_BODY_FRAGMENT_HEADING_PATTERNS:
        if pattern.search(normalized):
            return True, "context_body_fragment_heading"
    return False, ""


def match_running_header_footer_with_context(text: str, context: CleaningContext) -> tuple[bool, str]:
    """Return running header/footer matches, including conservative position hints."""
    matched, rule_id = match_running_header_footer(text)
    if matched:
        return True, rule_id
    normalized = _headingless(text)
    if not normalized or not _is_marginal_or_header_position(context):
        return False, ""
    if len(normalized.split()) <= 8 and re.search(r"\b(?:vol\.|no\.|journal|biotechnol|bioeng|open access)\b", normalized, re.I):
        return True, "context_running_header_by_position"
    return False, ""


def is_false_heading_with_context(text: str, context: CleaningContext) -> tuple[bool, str]:
    """Return whether a heading candidate is false using shared and contextual rules."""
    matched, rule_id = looks_like_reference_entry_with_context(text, context)
    if matched:
        return True, rule_id
    matched, rule_id = is_false_heading_candidate(text)
    if matched:
        return True, rule_id
    for matcher in (
        looks_like_body_fragment_heading,
        looks_like_back_matter_metadata,
        looks_like_affiliation_or_address,
    ):
        matched, rule_id = matcher(text, context)
        if matched:
            return True, rule_id
    return False, ""


def match_reference_noise(text: str) -> tuple[bool, str]:
    """Return whether text is a References-like section heading and its rule_id."""
    normalized = _headingless(text)
    if REFERENCE_SECTION_RE.match(normalized):
        return True, "reference_section"
    return False, ""


def match_contamination(text: str) -> tuple[bool, str]:
    """Return whether evidence text is contaminated by known non-evidence noise."""
    normalized = normalize_cleaning_text(text)
    for pattern, rule_id in CONTAMINATION_PATTERNS:
        if pattern.search(normalized):
            return True, rule_id
    running_header, rule_id = match_running_header_footer(normalized)
    if running_header:
        return True, rule_id
    return False, ""


def is_false_heading_candidate(text: str) -> tuple[bool, str]:
    """Return whether a heading candidate is likely not a real heading."""
    normalized = normalize_cleaning_text(text)
    for pattern, rule_id in FALSE_HEADING_PATTERNS:
        if pattern.search(normalized):
            return True, rule_id
    metadata_match, metadata_rule_id = match_metadata_noise(normalized)
    if metadata_match:
        return True, metadata_rule_id
    return False, ""


def classify_noise_rule(text: str) -> tuple[bool, str]:
    """Classify text against shared noise rules and return the first rule_id."""
    for matcher in (
        match_journal_preproof_noise,
        match_metadata_noise,
        match_running_header_footer,
        match_reference_noise,
        match_contamination,
        is_false_heading_candidate,
    ):
        matched, rule_id = matcher(text)
        if matched:
            return True, rule_id
    return False, ""


def classify_noise_rule_with_context(text: str, context: CleaningContext) -> tuple[bool, str]:
    """Classify text against shared rules plus conservative context-aware rules."""
    for matcher in (
        looks_like_reference_entry_with_context,
        looks_like_affiliation_or_address,
        looks_like_back_matter_metadata,
        match_running_header_footer_with_context,
        is_false_heading_with_context,
    ):
        matched, rule_id = matcher(text, context)
        if matched:
            return True, rule_id
    matched, rule_id = classify_noise_rule(text)
    if matched:
        return True, rule_id
    return False, ""
