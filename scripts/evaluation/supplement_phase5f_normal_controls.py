#!/usr/bin/env python3
"""Phase 5F-2B normal-control supplement and review.

This script is intentionally read-only with respect to retrieval, cleaning, and
eval logic. It reads existing Phase 5F-2 review outputs plus chunk JSONL files,
then writes supplement reports under reports/phase5f_normal_eval_quality_supplement.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE5F_DIR = ROOT / "reports/phase5f_normal_eval_quality"
OUT_DIR = ROOT / "reports/phase5f_normal_eval_quality_supplement"

GOOD_NORMAL_PATH = PHASE5F_DIR / "good_normal_control_candidates.jsonl"
SIGNOFF_PATH = PHASE5F_DIR / "normal_control_signoff.csv"
STATS_PATH = PHASE5F_DIR / "normal_quality_stats.json"
SUMMARY_PATH = PHASE5F_DIR / "summary.md"
DIAGNOSTIC_PATH = PHASE5F_DIR / "diagnostic_normal_controls.jsonl"

CHUNK_SOURCES = [
    Path("/tmp/biorag_phase4d_compact_chunks/chunks.jsonl"),
    Path("/tmp/biorag_phase5c4_full_enhanced/chunks/chunks.jsonl"),
    Path("/tmp/biorag_phase5d3_caption_cleanup/chunks/chunks.jsonl"),
]

PREFERRED_CHUNK_SOURCE = Path("/tmp/biorag_phase4d_compact_chunks/chunks.jsonl")

BACK_MATTER_RE = re.compile(
    r"(reference|bibliograph|acknowledg|funding|author contribution|credit authorship|"
    r"conflict|competing interest|correspondence|data availability|ethical approval|"
    r"abbreviation|supplementary|appendix)",
    re.IGNORECASE,
)
TABLE_FIGURE_RE = re.compile(
    r"\b(table|tables|figure|figures|fig\.|fig|supplementary table|supplementary figure|"
    r"primer|plasmid|strain and plasmid|oligonucleotide)\b",
    re.IGNORECASE,
)
TABLE_LIST_SHAPE_RE = re.compile(
    r"(\bGH\d+|\b[A-Z]{1,4}\d{2,}[A-Z0-9.-]*\b|"
    r"\b\d+(?:\.\d+)?\s*(?:rpm|h|min|mg/L|g/L|mM|uL|ml|mmol|%)\b)",
    re.IGNORECASE,
)
CITATION_RE = re.compile(r"\([A-Z][A-Za-z-]+(?: et al\.)?,? \d{4}[a-z]?(?:;[^)]*)?\)")
REFERENCE_LINE_RE = re.compile(
    r"(\b\d{1,4}:\d{2,4}[–-]\d{2,4}\b|"
    r"\b(?:Journal|J\.?|Appl|Microbiol|Biotechnol|Biochim|Biophys|Proc Natl Acad Sci)\b|"
    r"^[A-Z][A-Za-z-]+ [A-Z](?:,|\s+\(\d{4}\)))",
    re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
WHITESPACE_RE = re.compile(r"\s+")

EXCLUDED_BLOCK_TYPES = {
    "table_caption",
    "figure_caption",
    "table_text",
    "table",
    "figure",
    "title",
    "metadata",
    "references",
    "reference",
    "image",
    "noise",
}

QUERY_RULES = [
    "results_showed",
    "study_evaluated",
    "study_investigated",
    "method_used",
    "effect_observed",
    "comparison",
    "mechanism_role",
    "condition_response",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: stringify(row.get(field, "")) for field in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def preview(text: Any, limit: int = 420) -> str:
    compact = WHITESPACE_RE.sub(" ", str(text or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def normalize_text(text: str) -> str:
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"([A-Za-z])-+\s+([A-Za-z])", r"\1\2", text)
    text = re.sub(r"\bre\s+sults\b", "results", text, flags=re.I)
    text = re.sub(r"\bre\s+sponses\b", "responses", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def stable_blocks(row: dict[str, Any]) -> list[str]:
    blocks = row.get("stable_target_block_ids") or row.get("source_block_ids") or row.get("block_ids") or []
    if isinstance(blocks, str):
        return [item for item in blocks.split(";") if item]
    return [str(item) for item in blocks if item]


def ratio_ascii(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for char in text if ord(char) < 128) / len(text)


def digit_token_ratio(text: str) -> float:
    tokens = re.findall(r"\b\S+\b", text)
    if not tokens:
        return 1.0
    return sum(1 for token in tokens if any(ch.isdigit() for ch in token)) / len(tokens)


def punctuation_density(text: str, chars: str) -> float:
    if not text:
        return 0.0
    return sum(text.count(char) for char in chars) / len(text)


def load_chunk_index() -> tuple[dict[str, dict[str, Any]], dict[str, str], list[str]]:
    index: dict[str, dict[str, Any]] = {}
    source_by_chunk: dict[str, str] = {}
    missing: list[str] = []
    for source in CHUNK_SOURCES:
        if not source.exists():
            missing.append(str(source))
            continue
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                chunk_id = row.get("chunk_id")
                if chunk_id and chunk_id not in index:
                    index[chunk_id] = row
                    source_by_chunk[chunk_id] = str(source)
    return index, source_by_chunk, missing


def load_preferred_chunks() -> tuple[list[dict[str, Any]], str, list[str]]:
    missing: list[str] = []
    source = PREFERRED_CHUNK_SOURCE
    if not source.exists():
        missing.append(str(source))
        for fallback in CHUNK_SOURCES:
            if fallback.exists():
                source = fallback
                break
    if not source.exists():
        return [], "", missing + [str(item) for item in CHUNK_SOURCES if not item.exists()]

    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows, str(source), missing


def is_bad_section(section: str) -> bool:
    section = section.strip()
    if not section or section.lower() in {"title", "unknown"}:
        return True
    return bool(BACK_MATTER_RE.search(section))


def has_excluded_flags(row: dict[str, Any]) -> bool:
    return any(
        bool(row.get(flag))
        for flag in [
            "contains_figure_caption",
            "contains_table_caption",
            "contains_table_text",
            "contains_references",
            "contains_metadata",
            "contains_noise",
            "contains_image",
        ]
    )


def chunk_risk_flags(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    text = normalize_text(row.get("text", ""))
    section = str(row.get("section") or "")
    block_types = set(row.get("block_types") or [])
    evidence_types = set(row.get("evidence_types") or [])
    blocks = stable_blocks(row)

    if is_bad_section(section):
        flags.append("bad_section")
    if block_types != {"paragraph"}:
        flags.append("not_paragraph_only")
    if evidence_types != {"paragraph"}:
        flags.append("non_paragraph_evidence")
    if block_types & EXCLUDED_BLOCK_TYPES or evidence_types & EXCLUDED_BLOCK_TYPES:
        flags.append("excluded_block_type")
    if has_excluded_flags(row):
        flags.append("caption_table_reference_or_metadata_flag")
    if not blocks:
        flags.append("missing_source_block_ids")
    if len(text) < 350:
        flags.append("target_too_short")
    if len(text) > 2800:
        flags.append("target_too_long")
    if ratio_ascii(text) < 0.92:
        flags.append("non_english_or_encoding_risk")
    if digit_token_ratio(text) > 0.09:
        flags.append("digit_or_id_heavy")
    if punctuation_density(text, ";|") > 0.012:
        flags.append("list_like_punctuation")
    if TABLE_FIGURE_RE.search(section) or TABLE_FIGURE_RE.search(text[:700]):
        flags.append("table_or_figure_anchor")
    if TABLE_LIST_SHAPE_RE.search(text[:900]) and digit_token_ratio(text[:900]) > 0.06:
        flags.append("table_list_shape")
    if REFERENCE_LINE_RE.search(text[:900]):
        flags.append("reference_like_text")
    if re.search(r"https?://|www\.", text[:900], re.I):
        flags.append("url_or_resource_listing")
    if text.count("\n") > 10:
        flags.append("many_line_breaks")
    return flags


def clean_sentence(sentence: str) -> str:
    sentence = normalize_text(sentence)
    sentence = re.sub(r"^\W+", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip()


def candidate_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    pieces = SENTENCE_SPLIT_RE.split(text)
    result: list[str] = []
    for piece in pieces:
        sentence = clean_sentence(piece)
        if not sentence.endswith((".", "?", "!")):
            continue
        if not sentence[0].isupper():
            continue
        if re.match(r"^(With|And|Or|But|Although|Because|For example|For instance|In addition to|Along with|Required for|Drug that|A carbon source|The result has)\b", sentence):
            continue
        if len(sentence) < 90 or len(sentence) > 430:
            continue
        if digit_token_ratio(sentence) > 0.09:
            continue
        if punctuation_density(sentence, ";|") > 0.015:
            continue
        if TABLE_FIGURE_RE.search(sentence):
            continue
        if BACK_MATTER_RE.search(sentence):
            continue
        if re.search(r"\b(previous|anecdotally|resource|database|archive)\b", sentence, re.I):
            continue
        if REFERENCE_LINE_RE.search(sentence):
            continue
        if re.search(r"https?://|www\.", sentence, re.I):
            continue
        if len(CITATION_RE.findall(sentence)) > 3:
            continue
        if re.match(r"^[A-Z][A-Za-z ]{3,70}:\s+[A-Z]", sentence):
            continue
        result.append(sentence)
    return result


def short_topic(text: str, max_words: int = 8) -> str:
    text = re.sub(CITATION_RE, "", text)
    text = re.sub(r"\([^)]{1,45}\)", "", text)
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", text)
    stop = {
        "the",
        "and",
        "or",
        "with",
        "from",
        "that",
        "this",
        "these",
        "those",
        "their",
        "were",
        "was",
        "are",
        "for",
        "into",
        "using",
        "used",
        "study",
        "results",
        "showed",
        "demonstrated",
        "suggested",
        "indicated",
    }
    kept: list[str] = []
    for word in words:
        if word.lower() in stop:
            continue
        kept.append(word)
        if len(kept) >= max_words:
            break
    return " ".join(kept)


def compact_phrase(text: str, max_words: int = 13) -> str:
    text = normalize_text(text)
    text = re.sub(CITATION_RE, "", text)
    text = re.sub(r"\([^)]{1,70}\)", "", text)
    text = re.sub(
        r"^(The|A|An|These|Those|This|That|Moreover|Furthermore|Interestingly|Secondly|Overall|Here),?\s+",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"^(our|the)\s+results?\s+(?:suggest|show|demonstrate|indicate|reveal)\s+that\s+", "", text, flags=re.I)
    text = re.split(r"\s+(?:due to|because|whereas|while)\s+", text, maxsplit=1, flags=re.I)[0]
    text = text.strip(" ,.;:")
    words = text.split()
    text = " ".join(words[:max_words])
    text = re.sub(r"\s+(and|or|the|a|an|of|to|with|for)$", "", text, flags=re.I)
    return text.strip(" ,.;:")


def subject_before_verb(text: str) -> str:
    text = compact_phrase(text, 16)
    parts = re.split(
        r"\s+(?:was|were|is|are|can|could|may|might|would|resulted|led|increased|decreased|reduced|enhanced|improved|promoted|inhibited|showed|demonstrated|suggested|indicated|revealed)\b",
        text,
        maxsplit=1,
        flags=re.I,
    )
    return compact_phrase(parts[0], 10)


def strip_leading_context(sentence: str) -> str:
    sentence = re.sub(
        r"^(In this study|In the present study|Here,|Overall,|Moreover,|Furthermore|Interestingly|Secondly),?\s+",
        "",
        sentence,
        flags=re.I,
    )
    return sentence


def make_query(sentence: str, section: str) -> tuple[str, str] | None:
    sentence = clean_sentence(sentence)
    body = strip_leading_context(sentence).rstrip(".")
    lower = body.lower()

    match = re.search(r"\b(?:our|the|these|present|current)?\s*(results?|findings?)\s+(showed|demonstrated|suggested|indicated|revealed)\s+that\s+(.+)", body, re.I)
    if match:
        tail = match.group(3)
        topic = subject_before_verb(tail) or compact_phrase(tail, 8)
        if topic:
            return f"What did the results show about {topic}?", "results_showed"

    match = re.search(r"\bthese findings\s+(suggested?|indicated|revealed)\s+that\s+(.+)", body, re.I)
    if match:
        tail = match.group(2)
        topic = subject_before_verb(tail) or compact_phrase(tail, 8)
        if topic:
            return f"What did the findings suggest about {topic}?", "mechanism_role"

    match = re.search(r"\b(?:this|the|present|current)\s+study\s+(?:was undertaken to\s+)?(evaluated|examined|investigated|assessed|explored|compared)\s+(.+)", body, re.I)
    if match:
        verb = match.group(1).lower()
        tail = match.group(2)
        topic = compact_phrase(tail, 12)
        if topic:
            if verb == "compared":
                return f"What did the study compare for {topic}?", "comparison"
            rule = "study_evaluated" if verb in {"evaluated", "examined", "assessed"} else "study_investigated"
            return f"What did the study {verb} regarding {topic}?", rule

    match = re.search(r"\b(?:we|the authors)\s+(evaluated|examined|investigated|assessed|explored|compared)\s+(.+)", body, re.I)
    if match:
        verb = match.group(1).lower()
        topic = compact_phrase(match.group(2), 12)
        if topic:
            return f"What did the authors {verb} regarding {topic}?", "study_investigated"

    match = re.search(r"By\s+(.+?),\s+(.+?)\s+(?:can|could|may)\s+be\s+(improved|reduced|enhanced|increased|decreased)\b", body, re.I)
    if match:
        method = compact_phrase(match.group(1), 10)
        target = compact_phrase(match.group(2), 10)
        verb = match.group(3).lower()
        if method and target:
            return f"How can {target} be {verb}?", "method_used"

    match = re.search(r"(.+?)\s+(?:can|could|may|would)\s+be\s+(reduced|increased|decreased|enhanced|improved|inhibited|promoted|maintained|explained)\s+by\s+(.+)", body, re.I)
    if match and len(match.group(1)) < 160:
        target = compact_phrase(match.group(1), 11)
        verb = match.group(2).lower()
        if target:
            return f"How can {target} be {verb}?", "effect_observed"

    match = re.search(r"(One-way ANOVA|ANOVA|Tukey[^,.;]*)\s+(?:was|were)\s+used to\s+(.+)", body, re.I)
    if match:
        method = compact_phrase(match.group(1), 8)
        purpose = compact_phrase(match.group(2), 12)
        if method and purpose:
            return f"What was {method} used to compare?", "method_used"

    match = re.search(r"(.+?)\s+(?:was|were)\s+used to\s+(.+)", body, re.I)
    if match and len(match.group(1)) < 150:
        method = compact_phrase(match.group(1), 8)
        purpose = compact_phrase(match.group(2), 18)
        if method and purpose:
            return f"What was {method} used to {purpose}?", "method_used"

    match = re.search(r"For\s+.+?,\s+(.+?)\s+(?:was|were)\s+performed\s+(?:prior to|before)\s+(.+?)\s+in order to\s+(.+)", body, re.I)
    if match and len(match.group(1)) < 150:
        method = compact_phrase(match.group(1), 8)
        purpose = compact_phrase(match.group(3), 10)
        if method and purpose:
            return f"Why was {method} performed?", "method_used"

    match = re.search(r"(.+?)\s+(?:was|were)\s+performed\s+(?:prior to|before)?\s*(.+?)\s+in order to\s+(.+)", body, re.I)
    if match and len(match.group(1)) < 150:
        method = compact_phrase(match.group(1), 8)
        purpose = compact_phrase(match.group(3), 10)
        if method and purpose:
            return f"Why was {method} performed?", "method_used"

    match = re.search(r"(.+?)\s+(?:would|can|could|may)\s+be\s+indicated by\s+(.+)", body, re.I)
    if match and len(match.group(1)) < 160:
        marker = compact_phrase(match.group(1), 8)
        evidence = compact_phrase(match.group(2), 10)
        if marker and evidence:
            return f"What would indicate {marker}?", "condition_response"

    match = re.search(r"As\s+(.+?)\s+(leads to|can lead to|may lead to|could lead to)\s+(.+)", body, re.I)
    if match and len(match.group(1)) < 160:
        actor = compact_phrase(match.group(1), 8)
        if actor:
            return f"What does {actor} lead to?", "effect_observed"

    verb_base = {
        "resulted in": "result in",
        "led to": "lead to",
        "caused": "cause",
        "increased": "increase",
        "decreased": "decrease",
        "reduced": "reduce",
        "enhanced": "enhance",
        "improved": "improve",
        "promoted": "promote",
        "inhibited": "inhibit",
        "mitigated": "mitigate",
    }

    match = re.search(r"(.+?)\s+(indicated|suggested|revealed)\s+(.+)", body, re.I)
    if match and len(match.group(1)) < 160:
        actor = compact_phrase(match.group(1), 10)
        action = match.group(2).lower()
        if actor and actor.lower() not in {"the", "a", "an"}:
            return f"What did {actor} {action}?", "effect_observed"

    match = re.search(r"(.+?)\s+(?:may|can|could)\s+(result in|lead to|cause|increase|decrease|reduce|enhance|improve|promote|inhibit)\s+(.+)", body, re.I)
    if match and len(match.group(1)) < 180:
        actor = compact_phrase(match.group(1), 10)
        action = match.group(2).lower()
        if actor and " that " not in f" {actor.lower()} ":
            return f"What can {actor} {action}?", "effect_observed"

    match = re.search(r"(.+?)\s+(resulted in|led to|caused|increased|decreased|reduced|enhanced|improved|promoted|inhibited|mitigated)\s+(.+)", body, re.I)
    if match and len(match.group(1)) < 180:
        actor = compact_phrase(match.group(1), 10)
        action = verb_base.get(match.group(2).lower(), match.group(2).lower())
        if actor and " that " not in f" {actor.lower()} " and not re.search(r"\b(was reached|is a|are a)\b", actor, re.I):
            return f"What did {actor} {action}?", "effect_observed"

    match = re.search(r"(.+?)\s+(?:was|were)\s+(strongly\s+)?(up-regulated|down-regulated|increased|decreased|reduced|enhanced|improved)\b(.+)", body, re.I)
    if match and len(match.group(1)) < 180:
        target = compact_phrase(match.group(1), 10)
        verb = " ".join(part for part in [match.group(2) or "", match.group(3)] if part).strip().lower()
        if target:
            return f"How was {target} {verb}?", "condition_response"

    match = re.search(r"Compared with\s+(.+?),\s+(.+)", body, re.I)
    if match:
        baseline = compact_phrase(match.group(1), 8)
        topic = subject_before_verb(match.group(2)) or compact_phrase(match.group(2), 8)
        if baseline and topic:
            return f"How did {topic} differ compared with {baseline}?", "comparison"

    match = re.search(r"(.+?)\s+(?:can|could|may)\s+(act as|serve as|play|contribute to|regulate|promote|inhibit|maintain|support)\s+(.+)", body, re.I)
    if match and len(match.group(1)) < 160:
        actor = compact_phrase(match.group(1), 8)
        role = compact_phrase(match.group(3), 9)
        if actor and role:
            return f"What role can {actor} play in {role}?", "mechanism_role"

    match = re.search(r"(.+?)\s+(?:under|during|after|following)\s+(.+?)\s+(?:showed|displayed|exhibited|produced|accumulated|generated)\s+(.+)", body, re.I)
    if match:
        actor = compact_phrase(match.group(1), 8)
        condition = compact_phrase(match.group(2), 8)
        if actor and condition:
            return f"What response did {actor} show under {condition}?", "condition_response"

    return None


def query_risk_flags(query: str, target_preview: str) -> list[str]:
    flags: list[str] = []
    if not query:
        flags.append("missing_query")
    if len(query) < 35:
        flags.append("query_too_short")
    if re.search(r"^What did (In|While|When|By|The|A|An)\b", query):
        flags.append("query_bad_grammar")
    if re.search(
        r"^What did (After|Regardless|Several|Finally|Similar|Color|Applied|Secondary|Epidemiological|effect|results from|This|All HMO)\b",
        query,
    ):
        flags.append("query_bad_grammar")
    if re.search(r"\b(To do so|we first examined whether)\b", query, re.I):
        flags.append("query_bad_grammar")
    if re.search(r"\b(results? indicate that|findings suggest that|could be explained,|have associated|has associated)\b", query, re.I):
        flags.append("query_bad_grammar")
    if re.search(r"\b(led to|resulted in|reduced|increased|decreased|improved|enhanced|mitigated)\?$", query):
        flags.append("query_bad_grammar")
    if re.search(r"\b(revealed|indicated|suggested|increase|reduce|enhance|improve|explained|inhibited|reported|shown|cause)\?$", query):
        flags.append("query_bad_grammar")
    if re.search(r"\b(the|and|or|there)\?$", query, re.I):
        flags.append("query_bad_grammar")
    if re.match(r"what does (title|unknown|introduction|methods|results|discussion) report", query, re.I):
        flags.append("title_or_section_mechanical")
    if TABLE_FIGURE_RE.search(query):
        flags.append("table_or_figure_anchor")
    if digit_token_ratio(query) > 0.16:
        flags.append("query_digit_or_id_heavy")
    if normalize_text(query).lower().rstrip("?") in normalize_text(target_preview).lower():
        flags.append("query_copies_target")
    if not target_preview:
        flags.append("missing_target_preview")
    return flags


def quality_from_flags(chunk_flags: list[str], query_flags: list[str], blocks: list[str]) -> tuple[str, str, str, str]:
    if not blocks:
        return (
            "target_not_stable",
            "needs_manual_review",
            "manual_review",
            "Stable target block ids are missing, so the target is not stable enough for main evaluation.",
        )
    if "table_or_figure_anchor" in chunk_flags or "table_list_shape" in chunk_flags or "table_or_figure_anchor" in query_flags:
        return (
            "table_like_not_normal",
            "diagnostic_only",
            "move_to_diagnostic",
            "Target or query has table/list/figure-like anchors, so it should not enter the normal main denominator.",
        )
    if "title_or_section_mechanical" in query_flags or "query_copies_target" in query_flags:
        return (
            "title_derived_or_mechanical",
            "diagnostic_only",
            "rewrite_later",
            "Query is mechanical or copies the target too directly.",
        )
    severe = {
        "bad_section",
        "not_paragraph_only",
        "non_paragraph_evidence",
        "excluded_block_type",
        "caption_table_reference_or_metadata_flag",
        "target_too_short",
        "target_too_long",
        "non_english_or_encoding_risk",
        "digit_or_id_heavy",
        "list_like_punctuation",
        "reference_like_text",
        "url_or_resource_listing",
        "many_line_breaks",
    }
    if severe & set(chunk_flags):
        return (
            "exclude_from_main",
            "exclude_from_eval",
            "exclude",
            "Candidate failed conservative paragraph-normal filters.",
        )
    if query_flags:
        return (
            "needs_manual_review",
            "needs_manual_review",
            "manual_review",
            "Query has residual quality risks that need manual review before use.",
        )
    return (
        "good_normal_control",
        "main_eligible",
        "keep_main",
        "Paragraph-only non-table target with stable block ids and a natural deterministic query anchored in the target text.",
    )


def build_current_coverage(good_rows: list[dict[str, Any]], chunk_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source_phase = Counter(row.get("source_phase", "unknown") for row in good_rows)
    doc_counts = Counter(row.get("target_doc_id", "unknown") for row in good_rows)
    section_counts: Counter[str] = Counter()
    block_type_counts: Counter[str] = Counter()
    query_style_counts: Counter[str] = Counter()
    chunk_missing = 0

    for row in good_rows:
        chunk = chunk_index.get(row.get("target_chunk_id_candidate", ""))
        if not chunk:
            chunk_missing += 1
        section_counts[chunk.get("section", "unknown") if chunk else "unknown"] += 1
        block_types = chunk.get("block_types", []) if chunk else []
        block_type_counts[";".join(block_types) if block_types else "unknown"] += 1
        query = str(row.get("query", ""))
        if re.search(r"\bhow\b", query, re.I):
            query_style_counts["how"] += 1
        elif re.search(r"\bwhat result\b", query, re.I):
            query_style_counts["what_result"] += 1
        elif re.search(r"\bwhat (metabolic|engineering|strategy|pathway|enzyme|gene)\b", query, re.I):
            query_style_counts["what_domain_anchor"] += 1
        elif re.search(r"\bwhat\b", query, re.I):
            query_style_counts["what_general"] += 1
        else:
            query_style_counts["other"] += 1

    max_doc_count = max(doc_counts.values(), default=0)
    concentrated_docs = [doc for doc, count in doc_counts.items() if count >= 3]
    missing_to_30 = max(0, 30 - len(good_rows))
    return {
        "current_deduplicated_good_normal_count": len(good_rows),
        "missing_to_30": missing_to_30,
        "source_phase_distribution": dict(source_phase),
        "doc_id_distribution": dict(doc_counts),
        "section_distribution": dict(section_counts),
        "target_block_type_distribution": dict(block_type_counts),
        "query_style_distribution": dict(query_style_counts),
        "max_good_normal_per_doc": max_doc_count,
        "docs_with_3_or_more_good_normals": concentrated_docs,
        "chunk_metadata_missing_count": chunk_missing,
        "overconcentration_assessment": "no major concentration" if max_doc_count <= 2 else "review concentration",
    }


def write_current_coverage(coverage: dict[str, Any]) -> None:
    lines = [
        "# Phase 5F-2B Current Good Normal Coverage",
        "",
        f"- Current deduplicated good normal count: {coverage['current_deduplicated_good_normal_count']}",
        f"- Missing to reach 30: {coverage['missing_to_30']}",
        f"- Max good normal samples per doc: {coverage['max_good_normal_per_doc']}",
        f"- Overconcentration assessment: {coverage['overconcentration_assessment']}",
        f"- Chunk metadata missing count: {coverage['chunk_metadata_missing_count']}",
        "",
        "## Source Phase Distribution",
    ]
    for key, value in sorted(coverage["source_phase_distribution"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Doc Distribution"])
    for key, value in sorted(coverage["doc_id_distribution"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Section Distribution"])
    for key, value in sorted(coverage["section_distribution"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Target Block Types"])
    for key, value in sorted(coverage["target_block_type_distribution"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Query Style"])
    for key, value in sorted(coverage["query_style_distribution"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")
    write_md(OUT_DIR / "current_good_normal_coverage.md", "\n".join(lines))
    write_json(OUT_DIR / "current_good_normal_coverage.json", coverage)


def build_candidate_pool(
    chunks: list[dict[str, Any]],
    existing_doc_counts: Counter[str],
    existing_chunk_ids: set[str],
    chunk_source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_docs: set[str] = set()

    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "")
        doc_id = str(chunk.get("doc_id") or "")
        if not chunk_id or not doc_id or chunk_id in existing_chunk_ids:
            continue
        flags = chunk_risk_flags(chunk)
        if flags:
            continue
        sentences = candidate_sentences(chunk.get("text", ""))
        generated = None
        target_sentence = ""
        for sentence in sentences:
            generated = make_query(sentence, str(chunk.get("section") or ""))
            if generated:
                target_sentence = sentence
                break
        if not generated:
            continue

        section = str(chunk.get("section") or "")
        doc_penalty = existing_doc_counts.get(doc_id, 0)
        if doc_penalty >= 2:
            continue
        candidate_reason = "paragraph_only_non_table_non_reference_with_stable_blocks"
        if doc_id not in existing_doc_counts:
            candidate_reason += ";new_doc_relative_to_existing_good_normals"
        else:
            candidate_reason += ";low_existing_doc_overlap"

        rows.append(
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "section": section,
                "source_block_ids": stable_blocks(chunk),
                "block_types": chunk.get("block_types", []),
                "evidence_types": chunk.get("evidence_types", []),
                "text_preview": preview(target_sentence or chunk.get("text", ""), 520),
                "candidate_reason": candidate_reason,
                "risk_flags": [],
                "chunk_source": chunk_source,
                "selection_score": score_candidate(chunk, doc_penalty, doc_id in seen_docs),
                "query_rule_preview": generated[1],
            }
        )
        seen_docs.add(doc_id)

    rows.sort(key=lambda row: (-int(row["selection_score"]), int(row["doc_id"].split("_")[-1]), row["chunk_id"]))

    selected: list[dict[str, Any]] = []
    per_doc: Counter[str] = Counter()
    for row in rows:
        if per_doc[row["doc_id"]] >= 1:
            continue
        selected.append(row)
        per_doc[row["doc_id"]] += 1
        if len(selected) >= 80:
            break
    return selected


def score_candidate(chunk: dict[str, Any], doc_penalty: int, seen_doc: bool) -> int:
    section = str(chunk.get("section") or "").lower()
    score = 100
    if "result" in section or "discussion" in section:
        score += 30
    elif "method" in section or "material" in section:
        score += 15
    elif "introduction" in section:
        score += 8
    token_count = int(chunk.get("token_count") or 0)
    if 120 <= token_count <= 420:
        score += 15
    elif token_count > 650:
        score -= 15
    score -= doc_penalty * 20
    if seen_doc:
        score -= 10
    return score


def candidate_lookup(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("chunk_id")): row for row in chunks if row.get("chunk_id")}


def generate_queries(pool: list[dict[str, Any]], chunk_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    query_rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(pool[:45], start=1):
        chunk = chunk_by_id[candidate["chunk_id"]]
        target_sentence = ""
        generated = None
        for sentence in candidate_sentences(chunk.get("text", "")):
            generated = make_query(sentence, str(chunk.get("section") or ""))
            if generated:
                target_sentence = sentence
                break
        if not generated:
            continue
        query, rule = generated
        q_flags = query_risk_flags(query, target_sentence)
        c_flags = chunk_risk_flags(chunk)
        include_candidate = not c_flags and not q_flags
        query_rows.append(
            {
                "sample_id": f"phase5f2b_normal_supplement_{idx:04d}",
                "query_type": "normal_control",
                "query": query,
                "target_doc_id": chunk.get("doc_id"),
                "stable_target_block_ids": stable_blocks(chunk),
                "target_chunk_id_candidate": chunk.get("chunk_id"),
                "target_text_preview": preview(target_sentence, 620),
                "source_chunk_id": chunk.get("chunk_id"),
                "query_generation_rule": rule,
                "quality_risk_flags": sorted(set(c_flags + q_flags)),
                "include_in_main_denominator_candidate": include_candidate,
                "source_provenance": {
                    "source_phase": "Phase 5F-2B",
                    "source_file": candidate.get("chunk_source", ""),
                    "candidate_reason": candidate.get("candidate_reason", ""),
                },
            }
        )
    return query_rows


def signoff_queries(query_rows: list[dict[str, Any]], chunk_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_row in query_rows:
        chunk = chunk_by_id.get(str(query_row.get("source_chunk_id")), {})
        c_flags = chunk_risk_flags(chunk) if chunk else ["missing_chunk"]
        q_flags = query_risk_flags(str(query_row.get("query") or ""), str(query_row.get("target_text_preview") or ""))
        label, rec_label, action, rationale = quality_from_flags(
            sorted(set(c_flags)),
            sorted(set(q_flags)),
            stable_blocks(query_row),
        )
        rows.append(
            {
                "source_file": query_row.get("source_provenance", {}).get("source_file", ""),
                "phase": "Phase 5F-2B",
                "sample_id": query_row.get("sample_id"),
                "query": query_row.get("query"),
                "target_doc_id": query_row.get("target_doc_id"),
                "target_chunk_id": query_row.get("target_chunk_id_candidate"),
                "stable_target_block_ids": query_row.get("stable_target_block_ids"),
                "target_text_preview": query_row.get("target_text_preview"),
                "target_block_types": chunk.get("block_types", []),
                "target_evidence_types": chunk.get("evidence_types", []),
                "query_generation_rule": query_row.get("query_generation_rule"),
                "quality_risk_flags": sorted(set(c_flags + q_flags)),
                "quality_label": label,
                "recommended_label": rec_label,
                "recommended_action": action,
                "rationale": rationale,
                "risk_if_kept_in_main": ""
                if label == "good_normal_control"
                else "Would weaken the normal-control denominator or require manual correction before use.",
            }
        )
    return rows


def merge_good_normals(good_rows: list[dict[str, Any]], signoff_rows: list[dict[str, Any]], query_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    query_by_id = {row["sample_id"]: row for row in query_rows}
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    seen_text: set[tuple[str, str]] = set()

    for row in good_rows:
        key = (str(row.get("target_doc_id")), tuple(stable_blocks(row)))
        text_key = (
            normalize_text(str(row.get("query", ""))).lower(),
            normalize_text(str(row.get("target_text_preview", ""))).lower()[:240],
        )
        if key in seen:
            continue
        seen.add(key)
        if text_key[0] and text_key[1]:
            seen_text.add(text_key)
        merged_row = dict(row)
        merged_row["include_in_main_denominator"] = True
        merged_row.setdefault("source_provenance", {"source_phase": row.get("source_phase", "unknown"), "source_report": str(GOOD_NORMAL_PATH)})
        merged.append(merged_row)

    diagnostics: list[dict[str, Any]] = []
    for signoff in signoff_rows:
        query_row = dict(query_by_id.get(signoff["sample_id"], {}))
        if not query_row:
            continue
        query_row["quality_label"] = signoff["quality_label"]
        query_row["recommended_label"] = signoff["recommended_label"]
        query_row["recommended_action"] = signoff["recommended_action"]
        query_row["rationale"] = signoff["rationale"]
        if signoff["quality_label"] == "good_normal_control":
            key = (str(query_row.get("target_doc_id")), tuple(stable_blocks(query_row)))
            text_key = (
                normalize_text(str(query_row.get("query", ""))).lower(),
                normalize_text(str(query_row.get("target_text_preview", ""))).lower()[:240],
            )
            if key in seen or text_key in seen_text:
                continue
            seen.add(key)
            seen_text.add(text_key)
            query_row["include_in_main_denominator"] = True
            merged.append(query_row)
        else:
            query_row["include_in_main_denominator"] = False
            diagnostics.append(query_row)

    return merged, diagnostics


def write_merged_md(merged: list[dict[str, Any]], original_count: int, added_good_count: int) -> None:
    doc_counts = Counter(row.get("target_doc_id", "unknown") for row in merged)
    section_counts = Counter()
    for row in merged:
        provenance = row.get("source_provenance", {})
        if isinstance(provenance, dict) and provenance.get("section"):
            section_counts[provenance["section"]] += 1

    lines = [
        "# Phase 5F-2B Merged Good Normal Controls",
        "",
        f"- Original deduplicated good normal count: {original_count}",
        f"- Added good normal count: {added_good_count}",
        f"- Merged deduplicated good normal count: {len(merged)}",
        f"- Reached 30: {'yes' if len(merged) >= 30 else 'no'}",
        "",
        "## Merged Doc Distribution",
    ]
    for doc_id, count in sorted(doc_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {doc_id}: {count}")
    write_md(OUT_DIR / "good_normal_control_merged.md", "\n".join(lines))


def write_summary(stats: dict[str, Any]) -> None:
    lines = [
        "# Phase 5F-2B Normal Control Supplement Summary",
        "",
        f"1. Original deduplicated good normal count: {stats['original_good_normal_count']}",
        f"2. New candidate count: {stats['new_candidate_count']}",
        f"3. New good_normal_control count: {stats['new_good_normal_control_count']}",
        f"4. Merged deduplicated good normal count: {stats['merged_good_normal_count']}",
        f"5. Reached 30: {'yes' if stats['reached_30'] else 'no'}",
        f"6. Stable target coverage: {stats['stable_target_coverage']['with_stable_target_block_ids']}/{stats['stable_target_coverage']['total']} = {stats['stable_target_coverage']['coverage_rate']:.1%}",
        f"7. New samples mainly from docs/sections: {stats['new_sample_origin_summary']}",
        "8. Existing supplement exclusions: "
        f"table_like_not_normal={stats['supplement_quality_counts'].get('table_like_not_normal', 0)}, "
        f"title_derived_or_mechanical={stats['supplement_quality_counts'].get('title_derived_or_mechanical', 0)}, "
        f"query_target_mismatch={stats['supplement_quality_counts'].get('query_target_mismatch', 0)}",
        f"9. Recommend entering Phase 5F-3: {'yes' if stats['recommend_phase5f3'] else 'no'}",
        f"10. If below 30: {stats['if_below_30_recommendation']}",
        "11. Need to modify retrieval: no",
        "12. Need to rebuild index: no",
        "13. Need Qwen/RAGAS: no",
        "",
        "## Missing Inputs",
    ]
    if stats["missing_inputs"]:
        lines.extend(f"- {item}" for item in stats["missing_inputs"])
    else:
        lines.append("- none")
    write_md(OUT_DIR / "summary.md", "\n".join(lines))


def origin_summary(signoff_rows: list[dict[str, Any]]) -> str:
    good_rows = [row for row in signoff_rows if row["quality_label"] == "good_normal_control"]
    doc_counts = Counter(row.get("target_doc_id", "unknown") for row in good_rows)
    section_counts = Counter(row.get("target_section", "") for row in good_rows if row.get("target_section"))
    top_docs = ", ".join(f"{doc}:{count}" for doc, count in doc_counts.most_common(6))
    if not top_docs:
        top_docs = "none"
    top_sections = ", ".join(f"{section}:{count}" for section, count in section_counts.most_common(6))
    if not top_sections:
        top_sections = "sections recorded in candidate/signoff CSV"
    return f"docs [{top_docs}], sections [{top_sections}]"


def main() -> None:
    missing_inputs = [
        str(path)
        for path in [SUMMARY_PATH, SIGNOFF_PATH, GOOD_NORMAL_PATH, DIAGNOSTIC_PATH, STATS_PATH]
        if not path.exists()
    ]
    good_rows = read_jsonl(GOOD_NORMAL_PATH)
    signoff_previous = read_csv(SIGNOFF_PATH)

    chunk_index, _source_by_chunk, missing_chunk_sources = load_chunk_index()
    chunks, chunk_source, preferred_missing = load_preferred_chunks()
    missing_inputs.extend(missing_chunk_sources)
    missing_inputs.extend(preferred_missing)
    missing_inputs = sorted(set(missing_inputs))

    current_coverage = build_current_coverage(good_rows, chunk_index)
    write_current_coverage(current_coverage)

    existing_doc_counts = Counter(row.get("target_doc_id", "unknown") for row in good_rows)
    existing_chunk_ids = {str(row.get("target_chunk_id_candidate")) for row in good_rows if row.get("target_chunk_id_candidate")}
    pool = build_candidate_pool(chunks, existing_doc_counts, existing_chunk_ids, chunk_source)
    write_csv(
        OUT_DIR / "normal_candidate_pool.csv",
        pool,
        [
            "doc_id",
            "chunk_id",
            "section",
            "source_block_ids",
            "block_types",
            "evidence_types",
            "text_preview",
            "candidate_reason",
            "risk_flags",
            "chunk_source",
            "selection_score",
            "query_rule_preview",
        ],
    )

    chunk_by_id = candidate_lookup(chunks)
    query_rows = generate_queries(pool, chunk_by_id)
    write_jsonl(OUT_DIR / "normal_candidate_queries.jsonl", query_rows)

    signoff_rows = signoff_queries(query_rows, chunk_by_id)
    for row in signoff_rows:
        chunk = chunk_by_id.get(str(row.get("target_chunk_id")), {})
        row["target_section"] = chunk.get("section", "")
    write_csv(
        OUT_DIR / "supplement_normal_signoff.csv",
        signoff_rows,
        [
            "source_file",
            "phase",
            "sample_id",
            "query",
            "target_doc_id",
            "target_chunk_id",
            "target_section",
            "stable_target_block_ids",
            "target_text_preview",
            "target_block_types",
            "target_evidence_types",
            "query_generation_rule",
            "quality_risk_flags",
            "quality_label",
            "recommended_label",
            "recommended_action",
            "rationale",
            "risk_if_kept_in_main",
        ],
    )

    merged, diagnostics = merge_good_normals(good_rows, signoff_rows, query_rows)
    write_jsonl(OUT_DIR / "good_normal_control_merged.jsonl", merged)
    write_jsonl(OUT_DIR / "diagnostic_normal_supplement.jsonl", diagnostics)

    added_good_count = sum(1 for row in signoff_rows if row["quality_label"] == "good_normal_control")
    write_merged_md(merged, len(good_rows), added_good_count)

    with_stable = sum(1 for row in merged if stable_blocks(row))
    quality_counts = Counter(row["quality_label"] for row in signoff_rows)
    new_good_sections = Counter(row.get("target_section", "unknown") for row in signoff_rows if row["quality_label"] == "good_normal_control")
    new_good_docs = Counter(row.get("target_doc_id", "unknown") for row in signoff_rows if row["quality_label"] == "good_normal_control")
    stats = {
        "original_good_normal_count": len(good_rows),
        "phase5f2_signoff_reviewed_count": len(signoff_previous),
        "new_candidate_count": len(pool),
        "new_query_count": len(query_rows),
        "new_good_normal_control_count": added_good_count,
        "merged_good_normal_count": len(merged),
        "reached_30": len(merged) >= 30,
        "recommend_phase5f3": len(merged) >= 30 and with_stable == len(merged),
        "stable_target_coverage": {
            "total": len(merged),
            "with_stable_target_block_ids": with_stable,
            "without_stable_target_block_ids": len(merged) - with_stable,
            "coverage_rate": with_stable / len(merged) if merged else 0.0,
        },
        "supplement_quality_counts": dict(quality_counts),
        "new_good_doc_distribution": dict(new_good_docs),
        "new_good_section_distribution": dict(new_good_sections),
        "new_sample_origin_summary": origin_summary(signoff_rows),
        "if_below_30_recommendation": "Not applicable; merged good normal controls reached the minimum 30 target."
        if len(merged) >= 30
        else "Continue conservative manual curation from paragraph-only chunks; do not relax table/list/title-derived filters.",
        "missing_inputs": missing_inputs,
        "chunk_source_used_for_candidate_pool": chunk_source,
        "notes": [
            "No retrieval eval was run.",
            "No index was rebuilt.",
            "No Qwen, generation eval, RAGAS, OCR, or table extraction was invoked.",
        ],
    }
    write_json(OUT_DIR / "supplement_stats.json", stats)
    write_summary(stats)


if __name__ == "__main__":
    main()
