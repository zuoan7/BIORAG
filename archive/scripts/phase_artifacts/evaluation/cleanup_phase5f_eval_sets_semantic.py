#!/usr/bin/env python3
"""Phase 5F-3B semantic cleanup for main and diagnostic eval sets.

This script is intentionally self-contained. It reads the Phase 5F eval-set
artifacts and optional chunk JSONL files, then writes only semantic-cleanup
reports under reports/phase5f_eval_semantic_cleanup.

It does not run retrieval, generation, RAGAS, OCR, or index-building code.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "phase5f_eval_semantic_cleanup"

MAIN_PATH = ROOT / "reports" / "phase5f_clean_eval_set" / "clean_main_eval_set.jsonl"
DIAG_PATH = ROOT / "reports" / "phase5f_clean_eval_set" / "diagnostic_eval_set.jsonl"

REQUIRED_INPUTS = [
    ROOT / "reports" / "phase5f_clean_eval_set" / "clean_main_eval_set.jsonl",
    ROOT / "reports" / "phase5f_clean_eval_set" / "diagnostic_eval_set.jsonl",
    ROOT / "reports" / "phase5f_clean_eval_set" / "eval_quality_ledger.csv",
    ROOT / "reports" / "phase5f_clean_eval_set" / "target_mapping_audit.csv",
    ROOT / "reports" / "phase5f_clean_eval_set" / "summary.md",
    ROOT / "reports" / "phase5f_clean_eval_set" / "duplicate_overlap_report.md",
    ROOT
    / "reports"
    / "phase5f_normal_eval_quality_supplement"
    / "good_normal_control_merged.jsonl",
    ROOT
    / "reports"
    / "phase5f_eval_quality_audit"
    / "table_content_probe_audit.csv",
    ROOT
    / "reports"
    / "phase5f_eval_quality_audit"
    / "table_figure_probe_audit.csv",
    ROOT / "reports" / "phase5f_eval_quality_audit" / "normal_control_audit.csv",
    Path("/tmp/biorag_phase4d_compact_chunks/chunks.jsonl"),
    Path("/tmp/biorag_phase5c4_full_enhanced/chunks/chunks.jsonl"),
    Path("/tmp/biorag_phase5d3_caption_cleanup/chunks/chunks.jsonl"),
]

PREFERRED_CHUNK_PATHS = [
    Path("/tmp/biorag_phase5d3_caption_cleanup/chunks/chunks.jsonl"),
    Path("/tmp/biorag_phase5c4_full_enhanced/chunks/chunks.jsonl"),
    Path("/tmp/biorag_phase4d_compact_chunks/chunks.jsonl"),
]

MAIN_AUDIT_FIELDS = [
    "sample_id",
    "query_type",
    "original_query",
    "target_doc_id",
    "stable_target_block_ids",
    "target_text_preview",
    "source_phase",
    "ability_scope",
    "detected_issues",
    "semantic_quality_label",
    "recommended_action",
    "rewrite_candidate",
    "rewrite_confidence",
    "rationale",
]

REWRITE_FIELDS = [
    "sample_id",
    "query_type",
    "original_query",
    "rewritten_query",
    "target_doc_id",
    "stable_target_block_ids",
    "target_summary",
    "rewrite_reason",
    "rewrite_confidence",
    "risk_if_kept",
]

DIAG_LEDGER_FIELDS = [
    "sample_id",
    "query_type",
    "query",
    "target_doc_id",
    "stable_target_block_ids",
    "target_mapping_status",
    "diagnostic_label",
    "quality_label",
    "recommended_label",
    "assigned_bucket",
    "dedupe_key",
    "is_duplicate_query",
    "is_duplicate_target",
    "detected_issues",
    "rationale",
]

MANUAL_FIELDS = [
    "sample_id",
    "query_type",
    "original_query",
    "rewrite_candidate",
    "target_doc_id",
    "target_text_preview",
    "recommended_label",
    "rationale",
    "manual_question",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def stable_ids(row: dict[str, Any]) -> list[str]:
    value = row.get("stable_target_block_ids") or []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str) and value:
        return [value]
    return []


def stable_ids_s(row: dict[str, Any]) -> str:
    return "|".join(stable_ids(row))


def target_key(row: dict[str, Any]) -> str:
    ids = stable_ids_s(row)
    if ids:
        return f"{row.get('target_doc_id', '')}|{ids}"
    return f"{row.get('target_doc_id', '')}|{row.get('target_chunk_id_candidate', '')}"


def final_dedupe_key(row: dict[str, Any], query: str) -> str:
    caption_id = row.get("target_caption_block_id") or ""
    if caption_id:
        return f"{normalize_ws(query).lower()}|{row.get('target_doc_id', '')}|{caption_id}"
    return f"{normalize_ws(query).lower()}|{row.get('target_doc_id', '')}|{stable_ids_s(row)}"


def load_chunk_texts(rows: list[dict[str, Any]]) -> tuple[dict[str, str], list[str]]:
    wanted = {row.get("target_chunk_id_candidate") for row in rows}
    wanted = {str(v) for v in wanted if v}
    chunks: dict[str, str] = {}
    for path in PREFERRED_CHUNK_PATHS:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk_id = chunk.get("chunk_id")
                if chunk_id in wanted and chunk_id not in chunks:
                    chunks[chunk_id] = normalize_ws(chunk.get("text", ""))
        if wanted.issubset(chunks):
            break
    missing_inputs = [str(path) for path in REQUIRED_INPUTS if not path.exists()]
    return chunks, missing_inputs


def remove_caption_markers(text: str) -> str:
    text = normalize_ws(text)
    text = re.sub(r"^\[(?:TABLE|FIGURE)?\s*CAPTION\]\s*", "", text, flags=re.I)
    text = re.sub(r"\bCAPTION\b", "", text, flags=re.I)
    return normalize_ws(text)


def remove_table_figure_prefix(text: str) -> str:
    text = remove_caption_markers(text)
    text = re.sub(
        r"^((?:Table|TABLE|Fig\.?|Figure)\s+[S]?\d+)(?=[A-Z])",
        r"\1. ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"^(?:Table|TABLE|Fig\.?|Figure)\s+[S]?[A-Za-z0-9]+(?:[-.][A-Za-z0-9]+)?\s*"
        r"[:.|-]?\s*",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"^\(?continued\)?\.?\s*", "", text, flags=re.I)
    text = re.sub(r"^[A-Z]\.\s+", "", text)
    text = re.sub(r"\s+-\s+", "-", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    return normalize_ws(text)


def content_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9'/-]*|[0-9]+(?:\.[0-9]+)?", text or "")


def technical_token(token: str) -> bool:
    if re.search(r"\d", token):
        return True
    if any(ch in token for ch in ["/", "-", "_", "+", "′", "Δ", "α", "β"]):
        return True
    letters = [ch for ch in token if ch.isalpha()]
    if len(letters) >= 3:
        uppers = sum(1 for ch in letters if ch.isupper())
        if uppers / len(letters) > 0.45:
            return True
    return len(token) > 16


def issue_join(issues: list[str]) -> str:
    return "; ".join(dict.fromkeys(issues))


def strip_query_payload(query: str) -> str:
    q = normalize_ws(query).rstrip("?")
    patterns = [
        r"^Which table (?:contains|lists|reports)\s+(.+?)(?:\s+information|\s+measurements)?$",
        r"^Where are\s+(.+?)\s+summarized in a table$",
        r"^Which figure (?:shows|presents)\s+(?:the result for\s+)?(.+)$",
        r"^What does\s+(?:Table|Figure)\s+\S+\s+(?:report|show|describe)\s*$",
    ]
    for pattern in patterns:
        m = re.match(pattern, q, flags=re.I)
        if m and m.groups():
            return normalize_ws(m.group(1))
    return q


def query_issues(row: dict[str, Any]) -> list[str]:
    query = normalize_ws(row.get("query", ""))
    payload = strip_query_payload(query)
    tokens = content_words(payload)
    tech_count = sum(1 for t in tokens if technical_token(t))
    numeric_count = sum(1 for t in tokens if re.search(r"\d", t))
    issues: list[str] = []

    if re.search(r"\bCAPTION\b", query, flags=re.I):
        issues.append("caption_residue_in_query")
    if re.search(r"\b(?:TABLE|FIGURE)\s+[S]?[0-9A-Z]+", query):
        issues.append("table_or_figure_label_residue")
    if re.match(r"^Which table (?:contains|lists|reports)\b", query, flags=re.I):
        issues.append("mechanical_which_table_template")
    if re.match(r"^Where are .+ summarized in a table\??$", query, flags=re.I):
        issues.append("mechanical_where_table_template")
    if re.match(r"^Which figure (?:shows|presents)\b", query, flags=re.I):
        if len(tokens) >= 2 and tech_count / max(len(tokens), 1) >= 0.45:
            issues.append("mechanical_figure_caption_template")
    if re.match(r"^What does (?:Table|Figure)\s+\S+\s+(?:report|show|describe)", query, flags=re.I):
        issues.append("generic_table_or_figure_question")
    if len(tokens) >= 3 and tech_count / max(len(tokens), 1) >= 0.60:
        issues.append("token_soup_or_anchor_soup")
    if numeric_count >= 3 or (len(tokens) >= 3 and numeric_count / len(tokens) >= 0.45):
        issues.append("number_or_id_heavy_query")
    if re.search(r"\b(table cell|row|column|entry|cell value)\b", query, flags=re.I) and numeric_count:
        issues.append("row_cell_structured_table_query")
    if re.search(r"\b(OCR|image|photograph|micrograph|gel image)\b", query, flags=re.I):
        issues.append("ocr_or_image_query")
    return issues


def target_preview_looks_fragment(preview: str) -> bool:
    cleaned = remove_table_figure_prefix(preview)
    words = [w for w in content_words(cleaned) if w.lower() not in {"continued", "cont"}]
    raw = remove_caption_markers(preview)
    if re.search(r"\b(?:continued|cont\.)\b", raw, flags=re.I) and len(words) < 8:
        return True
    if re.match(
        r"^(?:Table|Fig\.?|Figure)\s+[S]?[A-Za-z0-9]+(?:[-.][A-Za-z0-9]+)?\.?\s*"
        r"(?:\([^)]+\)\.?)?$",
        raw,
        flags=re.I,
    ):
        return True
    if len(words) <= 2:
        return True
    return False


def target_summary(row: dict[str, Any], chunk_texts: dict[str, str]) -> tuple[str, bool]:
    preview = normalize_ws(row.get("target_text_preview", ""))
    chunk = chunk_texts.get(str(row.get("target_chunk_id_candidate") or ""), "")
    preview_fragment = target_preview_looks_fragment(preview)
    source = chunk if preview_fragment and chunk else preview
    cleaned = remove_table_figure_prefix(source)
    if not cleaned and chunk:
        cleaned = remove_table_figure_prefix(chunk)
    cleaned = re.split(r"\s{2,}", cleaned)[0]
    cleaned = normalize_ws(cleaned)
    if len(cleaned) > 260:
        cleaned = cleaned[:260].rsplit(" ", 1)[0]
    return cleaned, preview_fragment


def meaningful_target(summary: str) -> bool:
    words = [w for w in content_words(summary) if not re.fullmatch(r"\d+(?:\.\d+)?", w)]
    if len(words) < 5:
        return False
    if "continued" in summary.lower() and len(words) < 10:
        return False
    return True


def sentence_case_fragment(text: str) -> str:
    text = normalize_ws(text).strip(" .;:,")
    if not text:
        return text
    if len(text) > 1 and text[1] == ".":
        return text
    if len(text) > 1 and text[:2].isupper():
        return text
    return text[0].lower() + text[1:]


def short_subject(summary: str) -> str:
    subject = summary
    subject = re.split(r"\s+(?:strain|strains|source|description|characteristics)\s+", subject, maxsplit=1)[0]
    subject = normalize_ws(subject)
    if len(subject) > 220:
        subject = subject[:220].rsplit(" ", 1)[0]
    return sentence_case_fragment(subject)


def clean_query_subject(summary: str, max_words: int = 22) -> str:
    subject = short_subject(summary)
    subject = re.sub(r"\([^)]*\)", " ", subject)
    subject = re.sub(r"\[[^]]*\]", " ", subject)
    subject = re.sub(r"\b[A-E](?:[-,][A-E])?\.?\s+", " ", subject, flags=re.I)
    subject = re.sub(
        r"\b(?:strains/plasmids|strains or plasmids|strain genotype|characteristics source|description source)\b.*$",
        "",
        subject,
        flags=re.I,
    )
    subject = normalize_ws(subject)
    subject = re.sub(r"^(?:summary|list|overview|comparison) of\s+", "", subject, flags=re.I)
    subject = re.sub(r"^(?:showed|shows|showing) that\s+", "", subject, flags=re.I)
    subject = re.sub(r"\brespectively\b.*$", "", subject, flags=re.I)
    subject = re.sub(r"\.\s+(?:It|The|This|All)\b.*$", "", subject)
    subject = re.split(r"\.\s+(?=[A-Z#])", subject)[0]
    subject = re.sub(r"\s+([,.;:])", r"\1", subject)
    subject = re.sub(r"[()]+", " ", subject)
    subject = normalize_ws(subject).strip(" .;:,")
    words = subject.split()
    if len(words) > max_words:
        words = words[:max_words]
    while words and words[-1].strip(" .;:,").lower() in {"and", "or", "of", "the", "positive"}:
        words.pop()
    subject = " ".join(words).strip(" .;:,")
    return sentence_case_fragment(subject)


def query_with_subject(prefix: str, summary: str, fallback_subject: str) -> str:
    subject = clean_query_subject(summary)
    if len(content_words(subject)) < 5:
        subject = fallback_subject
    return f"{prefix} {subject}?"


def table_rewrite(summary: str) -> tuple[str, str, str]:
    s = summary.lower()
    if "proteins detected" in s and "culture media" in s:
        return (
            "Which table summarizes proteins detected from P. pastoris culture media under different fermentation conditions?",
            "high",
            "rewritten from table/caption semantics for detected culture-media proteins",
        )
    if "gff format" in s and "feature annotations" in s:
        return (
            "Which table summarizes feature annotations in GFF format for K. phaffii GS115?",
            "high",
            "rewritten from table/caption semantics for GFF feature annotations",
        )
    if "growth rate" in s and "csc" in s:
        return (
            "Which table compares growth rates for strains carrying different csc gene constructs?",
            "high",
            "rewritten from table/caption semantics for strain growth-rate comparisons",
        )
    if "gene disruptions" in s and "secretion" in s:
        return (
            "Which table summarizes how gene disruptions affected HyHEL-Fab secretion?",
            "high",
            "rewritten from table/caption semantics for secretion effects of gene disruptions",
        )
    if re.search(r"\bkinetic|enzyme|activity|selectivity|variant|wild-type|mutant\b", s):
        return (
            query_with_subject(
                "Which table reports",
                summary,
                "enzyme activity, selectivity, or kinetic parameters for the tested variants",
            ),
            "high",
            "rewritten from table/caption semantics for enzyme measurements",
        )
    if re.search(r"\bethanol|yield|production|productivity|biomass-specific|consumption|rates?\b", s):
        if "ethanol yield" in s and "glucose" in s and "xylose" in s:
            return (
                "Which table compares ethanol yields and glucose or xylose consumption rates?",
                "high",
                "rewritten from table/caption semantics for production and rate measurements",
            )
        if "ethanol yield" in s and "xylose" in s and "arabinose" in s:
            return (
                "Which table compares ethanol yields and xylose or arabinose consumption rates?",
                "high",
                "rewritten from table/caption semantics for production and rate measurements",
            )
        return (
            query_with_subject(
                "Which table compares",
                summary,
                "production yields or biomass-specific rates across the tested conditions",
            ),
            "high",
            "rewritten from table/caption semantics for production and rate measurements",
        )
    if re.search(r"\bfermentation|cultivation|culture|medium|methanol|shake-flask|bioreactor\b", s):
        return (
            query_with_subject(
                "Which table summarizes",
                summary,
                "fermentation or cultivation conditions and production results",
            ),
            "high",
            "rewritten from table/caption semantics for culture or fermentation conditions",
        )
    if re.search(r"\b(strains?|plasmids?|vectors?|construct(?:ed|ion)?|genotype)\b", s):
        if "strains and plasmids used" in s:
            return (
                "Which table summarizes the strains and plasmids used in this study?",
                "high",
                "rewritten from table/caption semantics for strains and plasmids",
            )
        if re.search(r"\bprimers?|oligonucleotides?\b", s):
            return (
                query_with_subject(
                    "Which table summarizes",
                    summary,
                    "the strains, plasmids, and primers used in the study",
                ),
                "high",
                "rewritten from table/caption semantics for strains, plasmids, or primers",
            )
        return (
            query_with_subject(
                "Which table summarizes",
                summary,
                "the strains, plasmids, or vectors used in the study",
            ),
            "high",
            "rewritten from table/caption semantics for strains, plasmids, or vectors",
        )
    if re.search(r"\bprimers?|oligonucleotides?|PCR\b", summary):
        return (
            query_with_subject(
                "Which table summarizes",
                summary,
                "the primers or oligonucleotides used in the study",
            ),
            "high",
            "rewritten from table/caption semantics for primer resources",
        )
    if re.search(r"\bgenes?|proteins?|proteomic|upregulated|expression|transcript|mRNA|ERAD|proteasomal\b", s):
        return (
            query_with_subject(
                "Which table summarizes",
                summary,
                "the genes or proteins measured under the study conditions",
            ),
            "high",
            "rewritten from table/caption semantics for gene or protein measurements",
        )
    if re.search(r"\bhuman milk|HMO|oligosaccharide|formula|infant|milk group|GBS\b", summary, flags=re.I):
        return (
            query_with_subject(
                "Which table summarizes",
                summary,
                "human milk oligosaccharide, infant formula, or milk-group measurements",
            ),
            "high",
            "rewritten from table/caption semantics for HMO or infant nutrition measurements",
        )
    if re.search(r"\bmicrobiota|microbial|bacterial|communities|lineages\b", s):
        return (
            query_with_subject(
                "Which table compares",
                summary,
                "microbial communities or bacterial lineages across the study groups",
            ),
            "high",
            "rewritten from table/caption semantics for microbial community comparisons",
        )
    if re.search(r"\bresponse surface|design|factor|optimization|parameters?\b", s):
        return (
            query_with_subject(
                "Which table reports",
                summary,
                "the experimental design or optimization results",
            ),
            "high",
            "rewritten from table/caption semantics for design or optimization results",
        )
    subject = clean_query_subject(summary)
    if len(content_words(subject)) >= 5:
        return (
            f"Which table summarizes {subject}?",
            "medium",
            "rewritten from the cleaned table/caption target summary",
        )
    return ("", "low", "target summary is too thin for a reliable natural rewrite")


def figure_rewrite(summary: str) -> tuple[str, str, str]:
    s = summary.lower()
    if "yield of acetate" in s and "fermentation" in s:
        return (
            "Which figure shows acetate yields during HMO fermentation with Bifidobacterium longum-dominant inocula?",
            "high",
            "rewritten from figure-caption semantics for acetate-yield fermentation results",
        )
    if re.search(r"\bsequence alignment|docking|conformations?\b", s):
        return (
            query_with_subject(
                "Which figure compares",
                summary,
                "sequence alignments or docking conformations for the enzymes",
            ),
            "high",
            "rewritten from figure-caption semantics for alignment or docking comparisons",
        )
    if re.search(r"\bgrowth curve|time course|fed-batch|fermentation|production\b", s):
        return (
            query_with_subject(
                "Which figure shows",
                summary,
                "growth or production over the fermentation time course",
            ),
            "high",
            "rewritten from figure-caption semantics for growth or production time courses",
        )
    if re.search(r"\bpathway|workflow|scheme|overview|biosynthesis\b", s):
        return (
            query_with_subject(
                "Which figure illustrates",
                summary,
                "the pathway or workflow used in the study",
            ),
            "high",
            "rewritten from figure-caption semantics for pathway or workflow figures",
        )
    if re.search(r"\beffect|impact|treatment|challenge|induction|condition|metal ions?\b", s):
        return (
            query_with_subject(
                "Which figure shows",
                summary,
                "the effect of the tested treatments or conditions",
            ),
            "high",
            "rewritten from figure-caption semantics for treatment effects",
        )
    if re.search(r"\bactivity|residual activity|enzyme|hydrolysis\b", s):
        return (
            query_with_subject(
                "Which figure compares",
                summary,
                "enzyme activity under the tested conditions",
            ),
            "high",
            "rewritten from figure-caption semantics for enzyme activity",
        )
    if re.search(r"\bMALDI|TOF|LC-MS|RP-HPLC|chromatography|mass spectrometry|SDS-PAGE\b", summary):
        return (
            query_with_subject(
                "Which figure presents",
                summary,
                "the analytical results used to characterize the product or protein",
            ),
            "high",
            "rewritten from figure-caption semantics for analytical characterization",
        )
    if re.search(r"\bmicrobiota|relative abundance|diversity|NMDS|Bray-Curtis\b", summary, flags=re.I):
        return (
            query_with_subject(
                "Which figure compares",
                summary,
                "microbiota composition or diversity across treatments",
            ),
            "high",
            "rewritten from figure-caption semantics for microbiota composition or diversity",
        )
    if re.search(r"\blocalization|subcellular|cells?|protein expression|cytokine\b", s):
        return (
            query_with_subject(
                "Which figure shows",
                summary,
                "cellular localization or expression responses in the tested cells",
            ),
            "high",
            "rewritten from figure-caption semantics for cellular or expression assays",
        )
    subject = clean_query_subject(summary)
    if len(content_words(subject)) >= 5:
        return (
            f"Which figure summarizes {subject}?",
            "medium",
            "rewritten from the cleaned figure-caption target summary",
        )
    return ("", "low", "target summary is too thin for a reliable natural rewrite")


def make_rewrite(row: dict[str, Any], chunk_texts: dict[str, str]) -> dict[str, str]:
    summary, preview_fragment = target_summary(row, chunk_texts)
    if not meaningful_target(summary):
        return {
            "rewrite_candidate": "",
            "rewrite_confidence": "low",
            "target_summary": summary,
            "rewrite_reason": "target summary is not semantically rich enough",
            "risk_if_kept": "would keep a token-soup or fragment-derived query in the main denominator",
            "preview_fragment": "true" if preview_fragment else "false",
        }

    if row.get("query_type") == "figure_caption":
        candidate, confidence, reason = figure_rewrite(summary)
    elif row.get("query_type") == "normal_control":
        candidate, confidence, reason = (
            normalize_ws(row.get("query", "")),
            "high",
            "normal-control query is already natural",
        )
    else:
        candidate, confidence, reason = table_rewrite(summary)

    if not candidate:
        confidence = "low"
    return {
        "rewrite_candidate": candidate,
        "rewrite_confidence": confidence,
        "target_summary": summary,
        "rewrite_reason": reason,
        "risk_if_kept": "mechanical or anchor-soup phrasing would leak into the strict main denominator",
        "preview_fragment": "true" if preview_fragment else "false",
    }


def classify_main(rows: list[dict[str, Any]], chunk_texts: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit: list[dict[str, Any]] = []
    for row in rows:
        issues = query_issues(row)
        rewrite = make_rewrite(row, chunk_texts)
        qtype = row.get("query_type", "")
        stable = stable_ids(row)
        preview_fragment = rewrite["preview_fragment"] == "true"
        action = "keep_strict_main"
        label = "main_strict"
        rationale: list[str] = []

        if not stable:
            issues.append("stable_target_block_ids_empty")
            action = "exclude_from_current_eval"
            label = "exclude_from_current_eval"
            rationale.append("Target cannot be placed in strict main without stable block ids.")
        elif any(issue in issues for issue in ["ocr_or_image_query", "row_cell_structured_table_query"]):
            action = "exclude_from_current_eval"
            label = "exclude_from_current_eval"
            rationale.append("Query requires OCR/image or row/cell structured-table capability outside current scope.")
        elif qtype == "normal_control":
            if issues:
                action = "manual_review"
                label = "needs_manual_review"
                rationale.append("Normal-control query has unexpected semantic issues and should be checked.")
            else:
                action = "keep_strict_main"
                label = "main_strict"
                rationale.append("Normal-control query is natural and target is stable.")
        elif preview_fragment:
            if row.get("target_associated_block_id"):
                action = "manual_review"
                label = "needs_manual_review"
                issues.append("target_preview_fragment_with_possible_associated_block")
                rationale.append("Target preview looks fragmentary, but an associated stable block may still be valid.")
            else:
                action = "move_to_diagnostic"
                label = "diagnostic_active"
                issues.append("target_preview_fragment_or_continued_caption")
                rationale.append("Fragment/continued caption should be monitored diagnostically rather than used in main.")
        elif rewrite["rewrite_confidence"] in {"high", "medium"} and rewrite["rewrite_candidate"]:
            if issues:
                action = "rewrite_and_keep_main"
                label = "main_rewritten"
                rationale.append("Original query is mechanical or anchor-derived, but target semantics support a natural rewrite.")
            else:
                action = "keep_strict_main"
                label = "main_strict"
                rationale.append("Query appears natural and target semantics are stable.")
        elif issues:
            action = "move_to_lexical_stress"
            label = "lexical_stress"
            rationale.append("Target has value, but the query is better suited for lexical stress than strict main.")
        else:
            action = "manual_review"
            label = "needs_manual_review"
            rationale.append("Automatic rules could not confidently decide whether this belongs in strict main.")

        audit.append(
            {
                "sample_id": row.get("sample_id", ""),
                "query_type": qtype,
                "original_query": normalize_ws(row.get("query", "")),
                "target_doc_id": row.get("target_doc_id", ""),
                "stable_target_block_ids": stable_ids_s(row),
                "target_text_preview": normalize_ws(row.get("target_text_preview", "")),
                "source_phase": row.get("source_phase", ""),
                "ability_scope": row.get("ability_scope", ""),
                "detected_issues": issue_join(issues),
                "semantic_quality_label": label,
                "recommended_action": action,
                "rewrite_candidate": rewrite["rewrite_candidate"],
                "rewrite_confidence": rewrite["rewrite_confidence"],
                "rationale": " ".join(rationale),
                "_row": row,
                "_target_summary": rewrite["target_summary"],
                "_rewrite_reason": rewrite["rewrite_reason"],
                "_risk_if_kept": rewrite["risk_if_kept"],
            }
        )

    seen_strict: set[str] = set()
    seen_query_doc: set[str] = set()
    for item in audit:
        if item["recommended_action"] not in {"keep_strict_main", "rewrite_and_keep_main"}:
            continue
        final_query = item["rewrite_candidate"] if item["recommended_action"] == "rewrite_and_keep_main" else item["original_query"]
        key = final_dedupe_key(item["_row"], final_query)
        query_doc_key = f"{normalize_ws(final_query).lower()}|{item['target_doc_id']}"
        if key in seen_strict and item["query_type"] != "normal_control":
            item["recommended_action"] = "move_to_lexical_stress"
            item["semantic_quality_label"] = "lexical_stress"
            item["detected_issues"] = issue_join(
                [part.strip() for part in item["detected_issues"].split(";") if part.strip()]
                + ["duplicate_rewritten_query_same_caption"]
            )
            item["rationale"] = (
                "Duplicate rewritten query for the same document/caption target; keep as lexical stress, not main."
            )
        elif query_doc_key in seen_query_doc:
            item["detected_issues"] = issue_join(
                [part.strip() for part in item["detected_issues"].split(";") if part.strip()]
                + ["duplicate_final_query_same_doc"]
            )
            if item["query_type"] == "normal_control":
                item["recommended_action"] = "manual_review"
                item["semantic_quality_label"] = "needs_manual_review"
                item["rationale"] = (
                    "Duplicate natural query for the same document with different target blocks; needs target-uniqueness review."
                )
            else:
                item["recommended_action"] = "move_to_lexical_stress"
                item["semantic_quality_label"] = "lexical_stress"
                item["rationale"] = (
                    "Duplicate final query for the same document after rewrite; keep out of strict main to avoid target ambiguity."
                )
        else:
            seen_strict.add(key)
            seen_query_doc.add(query_doc_key)

    rewrite_rows = []
    for item in audit:
        if item["recommended_action"] != "rewrite_and_keep_main":
            continue
        rewrite_rows.append(
            {
                "sample_id": item["sample_id"],
                "query_type": item["query_type"],
                "original_query": item["original_query"],
                "rewritten_query": item["rewrite_candidate"],
                "target_doc_id": item["target_doc_id"],
                "stable_target_block_ids": item["stable_target_block_ids"],
                "target_summary": item["_target_summary"],
                "rewrite_reason": item["_rewrite_reason"],
                "rewrite_confidence": item["rewrite_confidence"],
                "risk_if_kept": item["_risk_if_kept"],
            }
        )
    return audit, rewrite_rows


def strict_main_rows(audit: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_sample: set[str] = set()
    seen_target_query: set[str] = set()
    for item in audit:
        if item["recommended_action"] not in {"keep_strict_main", "rewrite_and_keep_main"}:
            continue
        row = item["_row"]
        query = item["rewrite_candidate"] if item["recommended_action"] == "rewrite_and_keep_main" else item["original_query"]
        strict_key = f"{normalize_ws(query).lower()}|{row.get('target_doc_id', '')}|{stable_ids_s(row)}"
        if item["sample_id"] in seen_sample or strict_key in seen_target_query:
            continue
        seen_sample.add(item["sample_id"])
        seen_target_query.add(strict_key)
        out = {
            "sample_id": item["sample_id"],
            "original_sample_id": item["sample_id"],
            "query_type": row.get("query_type", ""),
            "query": query,
            "original_query": item["original_query"],
            "query_was_rewritten": item["recommended_action"] == "rewrite_and_keep_main",
            "target_doc_id": row.get("target_doc_id", ""),
            "stable_target_block_ids": stable_ids(row),
            "target_caption_block_id": row.get("target_caption_block_id", ""),
            "target_associated_block_id": row.get("target_associated_block_id", ""),
            "target_chunk_id_candidate": row.get("target_chunk_id_candidate", ""),
            "source_phase": row.get("source_phase", ""),
            "source_file": row.get("source_file", ""),
            "ability_scope": row.get("ability_scope", ""),
            "quality_label": row.get("quality_label", ""),
            "semantic_quality_label": item["semantic_quality_label"],
            "include_in_main_denominator": True,
            "expected_capability": row.get("expected_capability", ""),
            "target_text_preview": normalize_ws(row.get("target_text_preview", "")),
            "rationale": item["rationale"],
            "notes": normalize_ws(row.get("notes", "")),
        }
        rows.append(out)
    return rows


def lexical_rows(audit: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in audit:
        if item["recommended_action"] != "move_to_lexical_stress":
            continue
        row = item["_row"]
        rows.append(
            {
                "sample_id": item["sample_id"],
                "original_sample_id": item["sample_id"],
                "query_type": row.get("query_type", ""),
                "query": item["original_query"],
                "original_query": item["original_query"],
                "include_in_main_denominator": False,
                "diagnostic_label": "lexical_stress",
                "target_doc_id": row.get("target_doc_id", ""),
                "stable_target_block_ids": stable_ids(row),
                "target_caption_block_id": row.get("target_caption_block_id", ""),
                "target_associated_block_id": row.get("target_associated_block_id", ""),
                "target_chunk_id_candidate": row.get("target_chunk_id_candidate", ""),
                "target_mapping_status": row.get("target_mapping_status", ""),
                "source_phase": row.get("source_phase", ""),
                "source_file": row.get("source_file", ""),
                "ability_scope": row.get("ability_scope", ""),
                "semantic_quality_label": "lexical_stress",
                "detected_issues": item["detected_issues"],
                "rewrite_candidate": item["rewrite_candidate"],
                "target_text_preview": item["target_text_preview"],
                "rationale": item["rationale"],
            }
        )
    return rows


def diagnostic_issues(row: dict[str, Any]) -> list[str]:
    issues = query_issues(row)
    if row.get("target_mapping_status") == "target_chunk_id_only":
        issues.append("target_chunk_id_only")
    if not stable_ids(row):
        issues.append("stable_target_block_ids_empty")
    if target_preview_looks_fragment(row.get("target_text_preview", "")):
        issues.append("target_preview_fragment_or_continued_caption")
    if row.get("diagnostic_label") in {"eval_only_noise", "generic_caption_query"}:
        issues.append("low_value_parser_or_generic_noise")
    return issues


def active_limit(label: str) -> int:
    return {
        "false_caption_noise": 30,
        "target_not_stable": 30,
        "low_confidence_table_association": 50,
        "retrieval_issue_candidate": 10_000,
        "title_derived_or_mechanical": 30,
        "table_like_not_normal": 30,
        "risk_slice": 30,
        "generic_caption_query": 10,
        "eval_only_noise": 10,
    }.get(label, 30)


def clean_diagnostic(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    query_counts = Counter(normalize_ws(row.get("query", "")).lower() for row in rows)
    target_counts = Counter(target_key(row) for row in rows)
    active_counts: Counter[str] = Counter()
    seen_exact: set[str] = set()
    active: list[dict[str, Any]] = []
    archive: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []

    for row in rows:
        label = row.get("diagnostic_label") or row.get("quality_label") or "unlabeled"
        exact_key = f"{normalize_ws(row.get('query', '')).lower()}|{target_key(row)}"
        issues = diagnostic_issues(row)
        duplicate_exact = exact_key in seen_exact
        seen_exact.add(exact_key)
        bucket = "archive"
        rationale = ""

        if (
            label == "needs_manual_review"
            or "target_preview_fragment_or_continued_caption" in issues
            and row.get("target_associated_block_id")
            or label in {"query_target_mismatch", "too_generic_or_ambiguous"}
        ):
            bucket = "manual_review"
            rationale = "Needs human judgment before it can be promoted, archived, or used diagnostically."
        elif label == "retrieval_issue_candidate":
            bucket = "active"
            rationale = "Retrieval issue candidates remain active as diagnostic probes."
        elif duplicate_exact:
            bucket = "archive"
            rationale = "Duplicate query/target diagnostic sample; archive to avoid overweighting repeated failures."
        elif active_counts[label] < active_limit(label):
            bucket = "active"
            rationale = f"Representative active sample for diagnostic label {label}."
        else:
            bucket = "archive"
            rationale = f"Over active limit for diagnostic label {label}; archived to keep diagnostics bounded."

        out = dict(row)
        out["include_in_main_denominator"] = False
        out["diagnostic_cleanup_bucket"] = bucket
        out["diagnostic_cleanup_rationale"] = rationale
        out["detected_issues"] = issue_join(issues)

        if bucket == "active":
            active_counts[label] += 1
            active.append(out)
        elif bucket == "manual_review":
            manual.append(out)
        else:
            archive.append(out)

        ledger.append(
            {
                "sample_id": row.get("sample_id", ""),
                "query_type": row.get("query_type", ""),
                "query": normalize_ws(row.get("query", "")),
                "target_doc_id": row.get("target_doc_id", ""),
                "stable_target_block_ids": stable_ids_s(row),
                "target_mapping_status": row.get("target_mapping_status", ""),
                "diagnostic_label": row.get("diagnostic_label", ""),
                "quality_label": row.get("quality_label", ""),
                "recommended_label": row.get("recommended_label", ""),
                "assigned_bucket": bucket,
                "dedupe_key": exact_key,
                "is_duplicate_query": query_counts[normalize_ws(row.get("query", "")).lower()] > 1,
                "is_duplicate_target": target_counts[target_key(row)] > 1,
                "detected_issues": issue_join(issues),
                "rationale": rationale,
            }
        )

    return active, archive, manual, ledger


def counter_md(counter: Counter[str]) -> str:
    if not counter:
        return "- none\n"
    return "\n".join(f"- {key or '(blank)'}: {value}" for key, value in counter.most_common()) + "\n"


def write_taxonomy() -> None:
    text = """# Phase 5F-3B Semantic Quality Taxonomy

This taxonomy is used only to clean eval-set semantics. It does not change
retrieval, cleaning, chunking, or eval logic.

## Labels

### main_strict

Can enter the strict main denominator. The query is natural, has a clear
target, has non-empty stable_target_block_ids, matches the target semantics,
has no CAPTION/TABLE/FIGURE label residue, is not token soup, is not a
number/ID pile, is not a mechanical template, does not require row/cell
structured table lookup, does not require OCR/image understanding, and is
something a real user might ask.

### main_rewritten

The original query is not acceptable as main, but the target is stable and
semantically clear. The query has been rewritten into a natural user question.
These samples can enter strict main only with original_query, rewrite_reason,
rewrite_confidence, and risk_if_kept recorded.

### lexical_stress

The target may be useful, but the query is a lexical stress probe rather than a
natural main-denominator question. Examples include "Which table contains X Y
Z", number-heavy table anchors, ID-heavy strain/plasmid anchors, and exact
table-text anchors. These do not enter strict main and should be reported as a
separate lexical stress set.

### diagnostic_active

Diagnostic probes that should remain available for Phase 5F-4 side-channel
statistics. Examples include representative false_caption_noise,
low_confidence_table_association, target_not_stable, parser artifact
monitoring, section metadata backlog, future structured-table/OCR/image
boundaries, and retrieval_issue_candidate.

### diagnostic_archive

Historically useful but not run by default. Examples include repeated false
caption samples, repeated generic Table 1 queries, repeated target_chunk_id_only
failures, duplicate query/target samples, and low-value legacy noise.

### needs_manual_review

Automatic judgment is insufficient. Examples include fragment-looking targets
with possibly valid associated blocks, queries that could be rewritten but
whose target uniqueness is uncertain, and repeated questions for the same target
that could pollute the denominator.

### exclude_from_current_eval

Outside the current capability boundary or semantically unusable. Examples
include OCR/image-only queries, row/cell structured-table-only queries, targets
that cannot be located, and queries that are not interpretable.

## Routing

- strict main: main_strict and main_rewritten only.
- lexical stress only: lexical_stress.
- diagnostic active: diagnostic_active only.
- archive: diagnostic_archive.
- human review first: needs_manual_review.
- not used in current eval: exclude_from_current_eval.

Samples are not deleted to improve metrics. Every downgraded sample keeps its
source, target, detected issue, and rationale so future phases can decide
whether to repair, archive, or evaluate it separately.
"""
    write_text(OUT_DIR / "semantic_quality_taxonomy.md", text)


def write_main_reports(audit: list[dict[str, Any]], rewrites: list[dict[str, Any]]) -> None:
    audit_rows = [{k: item.get(k, "") for k in MAIN_AUDIT_FIELDS} for item in audit]
    write_csv(OUT_DIR / "main_semantic_audit.csv", audit_rows, MAIN_AUDIT_FIELDS)
    write_csv(OUT_DIR / "query_rewrite_ledger.csv", rewrites, REWRITE_FIELDS)

    action_counts = Counter(item["recommended_action"] for item in audit)
    label_counts = Counter(item["semantic_quality_label"] for item in audit)
    issue_counts: Counter[str] = Counter()
    for item in audit:
        for issue in item["detected_issues"].split(";"):
            issue = issue.strip()
            if issue:
                issue_counts[issue] += 1

    examples = []
    for item in audit:
        if item["recommended_action"] in {"rewrite_and_keep_main", "move_to_lexical_stress", "manual_review"}:
            examples.append(
                f"- {item['sample_id']} ({item['query_type']}): {item['recommended_action']} - "
                f"{item['original_query']}"
            )
        if len(examples) >= 20:
            break

    main_md = f"""# Main Semantic Audit

## Counts

Original main samples: {len(audit)}

Recommended actions:
{counter_md(action_counts)}
Semantic labels:
{counter_md(label_counts)}
Detected issues:
{counter_md(issue_counts)}
## Representative Decisions

{chr(10).join(examples) if examples else '- none'}

The audit uses query/target semantic quality only. Historical hit/miss metrics
are not used for keep, rewrite, downgrade, or manual-review decisions.
"""
    write_text(OUT_DIR / "main_semantic_audit.md", main_md)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rewrites:
        grouped[row["query_type"]].append(row)
    parts = ["# Rewrite Examples", ""]
    for qtype, rows in grouped.items():
        parts.append(f"## {qtype}")
        for row in rows[:8]:
            parts.append("")
            parts.append(f"- sample_id: {row['sample_id']}")
            parts.append(f"  - original: {row['original_query']}")
            parts.append(f"  - rewritten: {row['rewritten_query']}")
            parts.append(f"  - reason: {row['rewrite_reason']}")
            parts.append(f"  - confidence: {row['rewrite_confidence']}")
    write_text(OUT_DIR / "rewrite_examples.md", "\n".join(parts))


def write_strict_reports(rows: list[dict[str, Any]]) -> None:
    write_jsonl(OUT_DIR / "strict_main_eval_set.jsonl", rows)
    qtype_counts = Counter(row.get("query_type", "") for row in rows)
    rewritten = sum(1 for row in rows if row.get("query_was_rewritten"))
    stable = sum(1 for row in rows if row.get("stable_target_block_ids"))
    caption_residue = sum(1 for row in rows if re.search(r"\bCAPTION\b", row.get("query", ""), re.I))
    text = f"""# Strict Main Eval Set Summary

- strict_main_count: {len(rows)}
- rewritten_count: {rewritten}
- stable_target_coverage: {stable}/{len(rows)}
- caption_residue_in_query: {caption_residue}

## Query Type Distribution

{counter_md(qtype_counts)}
All rows set include_in_main_denominator=true and exclude lexical_stress,
diagnostic-only, needs_manual_review, target_chunk_id_only, OCR/image, and
row/cell structured-table-only queries.
"""
    write_text(OUT_DIR / "strict_main_eval_set_summary.md", text)


def write_lexical_reports(rows: list[dict[str, Any]]) -> None:
    write_jsonl(OUT_DIR / "lexical_stress_eval_set.jsonl", rows)
    qtype_counts = Counter(row.get("query_type", "") for row in rows)
    issue_counts: Counter[str] = Counter()
    for row in rows:
        for issue in row.get("detected_issues", "").split(";"):
            issue = issue.strip()
            if issue:
                issue_counts[issue] += 1
    text = f"""# Lexical Stress Summary

- lexical_stress_count: {len(rows)}
- include_in_main_denominator: false for all rows
- diagnostic_label: lexical_stress for all rows

## Query Type Distribution

{counter_md(qtype_counts)}
## Issue Distribution

{counter_md(issue_counts)}
Lexical stress may be reported in Phase 5F-4 as a separate slice, but it must
not be mixed into the strict main denominator.
"""
    write_text(OUT_DIR / "lexical_stress_summary.md", text)


def write_diagnostic_reports(
    original: list[dict[str, Any]],
    active: list[dict[str, Any]],
    archive: list[dict[str, Any]],
    manual: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
) -> None:
    write_jsonl(OUT_DIR / "diagnostic_active_eval_set.jsonl", active)
    write_jsonl(OUT_DIR / "diagnostic_archive_eval_set.jsonl", archive)
    write_jsonl(OUT_DIR / "diagnostic_needs_manual_review.jsonl", manual)
    write_csv(OUT_DIR / "diagnostic_cleanup_ledger.csv", ledger, DIAG_LEDGER_FIELDS)

    query_counts = Counter(normalize_ws(row.get("query", "")).lower() for row in original)
    target_counts = Counter(target_key(row) for row in original)
    label_dist = Counter(row.get("diagnostic_label", "") for row in original)
    active_dist = Counter(row.get("diagnostic_label", "") for row in active)
    archive_dist = Counter(row.get("diagnostic_label", "") for row in archive)
    manual_dist = Counter(row.get("diagnostic_label", "") for row in manual)
    duplicate_query_rows = sum(1 for row in original if query_counts[normalize_ws(row.get("query", "")).lower()] > 1)
    duplicate_target_rows = sum(1 for row in original if target_counts[target_key(row)] > 1)
    target_chunk_only = sum(1 for row in original if row.get("target_mapping_status") == "target_chunk_id_only")
    stable = sum(1 for row in original if row.get("target_mapping_status") == "stable_target")

    text = f"""# Diagnostic Cleanup Summary

- original_diagnostic_total: {len(original)}
- diagnostic_active: {len(active)}
- diagnostic_archive: {len(archive)}
- diagnostic_manual_review: {len(manual)}
- duplicate_query_rows: {duplicate_query_rows}
- duplicate_target_rows: {duplicate_target_rows}
- target_chunk_id_only: {target_chunk_only}
- stable_target: {stable}

## Original Diagnostic Label Distribution

{counter_md(label_dist)}
## Active Label Distribution

{counter_md(active_dist)}
## Archive Label Distribution

{counter_md(archive_dist)}
## Manual Review Label Distribution

{counter_md(manual_dist)}
Active diagnostics are bounded representative probes. Archive rows are kept with
their source metadata but should not run by default.
"""
    write_text(OUT_DIR / "diagnostic_cleanup_summary.md", text)


def manual_question(row: dict[str, Any], origin: str) -> str:
    if origin == "main_manual":
        return "Is the associated stable block sufficient to make this a natural main sample, or should it stay out?"
    if origin == "main_medium_rewrite":
        return "Is the medium-confidence rewrite faithful and unique enough for strict main?"
    if origin == "main_lexical_boundary":
        return "Should this remain lexical stress, or can a better natural query preserve the target semantics?"
    if origin == "diagnostic_manual":
        return "Can this diagnostic sample be repaired or promoted, or should it remain diagnostic/archive?"
    return "Review semantic fit and target uniqueness."


def build_manual_pack(
    audit: list[dict[str, Any]],
    diag_manual: list[dict[str, Any]],
    diag_active: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[tuple[int, str, dict[str, Any]]] = []

    for item in audit:
        priority = None
        origin = ""
        if item["recommended_action"] == "manual_review":
            priority = 10
            origin = "main_manual"
        elif item["recommended_action"] == "rewrite_and_keep_main" and item["rewrite_confidence"] == "medium":
            priority = 20
            origin = "main_medium_rewrite"
        elif item["recommended_action"] == "move_to_lexical_stress" and item["query_type"] == "table_content":
            priority = 30
            origin = "main_lexical_boundary"
        elif item["target_doc_id"] == "doc_0367":
            priority = 5
            origin = "doc_0367_anchor"
        if priority is None:
            continue
        candidates.append(
            (
                priority,
                item["sample_id"],
                {
                    "sample_id": item["sample_id"],
                    "query_type": item["query_type"],
                    "original_query": item["original_query"],
                    "rewrite_candidate": item["rewrite_candidate"],
                    "target_doc_id": item["target_doc_id"],
                    "target_text_preview": item["target_text_preview"],
                    "recommended_label": item["semantic_quality_label"],
                    "rationale": item["rationale"],
                    "manual_question": manual_question(item, origin),
                },
            )
        )

    for row in diag_manual:
        candidates.append(
            (
                40,
                row.get("sample_id", ""),
                {
                    "sample_id": row.get("sample_id", ""),
                    "query_type": row.get("query_type", ""),
                    "original_query": normalize_ws(row.get("query", "")),
                    "rewrite_candidate": "",
                    "target_doc_id": row.get("target_doc_id", ""),
                    "target_text_preview": normalize_ws(row.get("target_text_preview", "")),
                    "recommended_label": row.get("diagnostic_label", "needs_manual_review"),
                    "rationale": row.get("diagnostic_cleanup_rationale", ""),
                    "manual_question": manual_question(row, "diagnostic_manual"),
                },
            )
        )

    for row in diag_active:
        if row.get("target_mapping_status") == "stable_target" and row.get("diagnostic_label") in {
            "low_confidence_table_association",
            "retrieval_issue_candidate",
        }:
            candidates.append(
                (
                    50,
                    row.get("sample_id", ""),
                    {
                        "sample_id": row.get("sample_id", ""),
                        "query_type": row.get("query_type", ""),
                        "original_query": normalize_ws(row.get("query", "")),
                        "rewrite_candidate": "",
                        "target_doc_id": row.get("target_doc_id", ""),
                        "target_text_preview": normalize_ws(row.get("target_text_preview", "")),
                        "recommended_label": row.get("diagnostic_label", ""),
                        "rationale": "Stable diagnostic candidate that may be promotable after human review.",
                        "manual_question": "Is this diagnostic sample promotable to strict main after rewriting?",
                    },
                )
            )

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for _, _, row in sorted(candidates, key=lambda item: (item[0], item[1])):
        key = row["sample_id"]
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
        if len(rows) >= 80:
            break
    return rows


def write_manual_pack(rows: list[dict[str, Any]]) -> None:
    write_csv(OUT_DIR / "manual_review_pack.csv", rows, MANUAL_FIELDS)
    parts = ["# Manual Review Pack", "", f"Total selected: {len(rows)}", ""]
    for row in rows:
        parts.append(f"## {row['sample_id']}")
        parts.append("")
        parts.append(f"- query_type: {row['query_type']}")
        parts.append(f"- original_query: {row['original_query']}")
        if row["rewrite_candidate"]:
            parts.append(f"- rewrite_candidate: {row['rewrite_candidate']}")
        parts.append(f"- target_doc_id: {row['target_doc_id']}")
        parts.append(f"- recommended_label: {row['recommended_label']}")
        parts.append(f"- rationale: {row['rationale']}")
        parts.append(f"- manual_question: {row['manual_question']}")
        parts.append(f"- target_text_preview: {row['target_text_preview'][:500]}")
        parts.append("")
    write_text(OUT_DIR / "manual_review_pack.md", "\n".join(parts))


def strict_gate_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    caption_residue = [
        row["sample_id"]
        for row in rows
        if re.search(r"\bCAPTION\b", row.get("query", ""), flags=re.I)
    ]
    target_chunk_only = [
        row["sample_id"]
        for row in rows
        if row.get("target_mapping_status") == "target_chunk_id_only"
    ]
    manual = [
        row["sample_id"]
        for row in rows
        if row.get("semantic_quality_label") == "needs_manual_review"
        or row.get("quality_label") == "needs_manual_review"
    ]
    lexical = [
        row["sample_id"]
        for row in rows
        if row.get("semantic_quality_label") == "lexical_stress"
        or row.get("diagnostic_label") == "lexical_stress"
    ]
    row_cell = []
    ocr_image = []
    for row in rows:
        issues = query_issues({"query": row.get("query", "")})
        if "row_cell_structured_table_query" in issues:
            row_cell.append(row["sample_id"])
        if "ocr_or_image_query" in issues:
            ocr_image.append(row["sample_id"])
    return {
        "stable_target_count": sum(1 for row in rows if row.get("stable_target_block_ids")),
        "caption_residue_count": len(caption_residue),
        "caption_residue_sample_ids": caption_residue,
        "target_chunk_id_only_count": len(target_chunk_only),
        "needs_manual_review_count": len(manual),
        "lexical_stress_count": len(lexical),
        "row_cell_structured_query_count": len(row_cell),
        "ocr_or_image_query_count": len(ocr_image),
    }


def write_global_reports(
    main_rows: list[dict[str, Any]],
    audit: list[dict[str, Any]],
    strict: list[dict[str, Any]],
    lexical: list[dict[str, Any]],
    diag_rows: list[dict[str, Any]],
    diag_active: list[dict[str, Any]],
    diag_archive: list[dict[str, Any]],
    diag_manual: list[dict[str, Any]],
    missing_inputs: list[str],
) -> None:
    action_counts = Counter(item["recommended_action"] for item in audit)
    strict_qtypes = Counter(row.get("query_type", "") for row in strict)
    rewritten_count = sum(1 for row in strict if row.get("query_was_rewritten"))
    gates = strict_gate_stats(strict)

    stats = {
        "missing_inputs": missing_inputs,
        "original_main_count": len(main_rows),
        "strict_main_count": len(strict),
        "strict_main_query_type_counts": dict(strict_qtypes),
        "rewritten_query_count": rewritten_count,
        "lexical_stress_count": len(lexical),
        "main_move_to_diagnostic_count": action_counts.get("move_to_diagnostic", 0),
        "main_manual_review_count": action_counts.get("manual_review", 0),
        "main_exclude_count": action_counts.get("exclude_from_current_eval", 0),
        "strict_main_stable_target_coverage": f"{gates['stable_target_count']}/{len(strict)}",
        "strict_main_gates": gates,
        "original_diagnostic_count": len(diag_rows),
        "diagnostic_active_count": len(diag_active),
        "diagnostic_archive_count": len(diag_archive),
        "diagnostic_manual_review_count": len(diag_manual),
        "diagnostic_active_label_counts": dict(Counter(row.get("diagnostic_label", "") for row in diag_active)),
        "diagnostic_archive_label_counts": dict(Counter(row.get("diagnostic_label", "") for row in diag_archive)),
        "diagnostic_manual_label_counts": dict(Counter(row.get("diagnostic_label", "") for row in diag_manual)),
        "recommend_enter_phase5f4": True,
        "retrieval_changes_needed": False,
        "rebuild_index_needed": False,
        "qwen_ragas_needed": False,
    }
    (OUT_DIR / "semantic_cleanup_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    before_after = f"""# Before/After Comparison

| Metric | Before | After |
| --- | ---: | ---: |
| Main denominator samples | {len(main_rows)} | {len(strict)} |
| Table content in main | {sum(1 for r in main_rows if r.get('query_type') == 'table_content')} | {strict_qtypes.get('table_content', 0)} |
| Caption-level table in main | {sum(1 for r in main_rows if r.get('query_type') == 'caption_level_table')} | {strict_qtypes.get('caption_level_table', 0)} |
| Figure caption in main | {sum(1 for r in main_rows if r.get('query_type') == 'figure_caption')} | {strict_qtypes.get('figure_caption', 0)} |
| Normal control in main | {sum(1 for r in main_rows if r.get('query_type') == 'normal_control')} | {strict_qtypes.get('normal_control', 0)} |
| Rewritten main queries | 0 | {rewritten_count} |
| Lexical stress samples | 0 | {len(lexical)} |
| Main manual-review samples | 0 | {action_counts.get('manual_review', 0)} |
| Diagnostic active samples | {len(diag_rows)} | {len(diag_active)} |
| Diagnostic archive samples | 0 | {len(diag_archive)} |
| Diagnostic manual-review samples | 0 | {len(diag_manual)} |

The old clean_main_eval_set should no longer be used as the formal main
denominator. Phase 5F-4 should use strict_main_eval_set for main metrics and
report lexical_stress and diagnostic_active separately.
"""
    write_text(OUT_DIR / "before_after_comparison.md", before_after)

    summary = f"""# Phase 5F-3B Semantic Cleanup Summary

1. Original main count: {len(main_rows)}
2. Strict main count: {len(strict)}
3. Strict main query_type counts: {dict(strict_qtypes)}
4. Rewritten query count: {rewritten_count}
5. Downgraded to lexical_stress: {len(lexical)}
6. Downgraded to diagnostic: {action_counts.get('move_to_diagnostic', 0)}
7. Main manual_review count: {action_counts.get('manual_review', 0)}
8. Strict main stable target coverage: {gates['stable_target_count']}/{len(strict)}
9. Strict main CAPTION residue count: {gates['caption_residue_count']}
10. Strict main target_chunk_id_only count: {gates['target_chunk_id_only_count']}
11. Strict main needs_manual_review count: {gates['needs_manual_review_count']}
12. Strict main row/cell/OCR/image query count: {gates['row_cell_structured_query_count'] + gates['ocr_or_image_query_count']}
13. Original diagnostic count: {len(diag_rows)}
14. diagnostic_active / archive / manual_review: {len(diag_active)} / {len(diag_archive)} / {len(diag_manual)}
15. Recommendation for Phase 5F-4: yes, use the cleaned semantic splits.
16. Phase 5F-4 main denominator: strict_main_eval_set.jsonl, not old clean_main_eval_set.jsonl.
17. lexical_stress and diagnostic_active: report separately, never mix with main.
18. Retrieval changes needed: no.
19. Rebuild index needed: no.
20. Qwen/RAGAS needed: no.

Missing optional/expected inputs recorded: {len(missing_inputs)}
"""
    write_text(OUT_DIR / "summary.md", summary)

    next_phase = """# Phase 5F-4 Plan

Phase 5F-4 should use:

- strict_main_eval_set.jsonl as the only main denominator.
- lexical_stress_eval_set.jsonl as a separate lexical stress slice.
- diagnostic_active_eval_set.jsonl as a separate diagnostic slice.
- diagnostic_archive_eval_set.jsonl as retained history, not run by default.
- diagnostic_needs_manual_review.jsonl only after human review.

Phase 5F-4 should not:

- Use the old clean_main_eval_set.jsonl as the formal main denominator.
- Mix lexical stress into main.
- Mix diagnostic active into main.
- Run the full diagnostic archive by default.
- Tune retrieval parameters.
- Change cleaning logic.
- Change BM25, dense, hybrid, or reranker behavior.
- Call Qwen.

No retrieval changes, index rebuild, generation eval, RAGAS, OCR, or parser/table
object changes are required for Phase 5F-4.
"""
    write_text(OUT_DIR / "next_phase_plan.md", next_phase)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_rows = read_jsonl(MAIN_PATH)
    diag_rows = read_jsonl(DIAG_PATH)
    chunk_texts, missing_inputs = load_chunk_texts(main_rows + diag_rows)

    write_taxonomy()
    audit, rewrites = classify_main(main_rows, chunk_texts)
    strict = strict_main_rows(audit)
    lexical = lexical_rows(audit)
    diag_active, diag_archive, diag_manual, diag_ledger = clean_diagnostic(diag_rows)
    manual_pack = build_manual_pack(audit, diag_manual, diag_active)

    write_main_reports(audit, rewrites)
    write_strict_reports(strict)
    write_lexical_reports(lexical)
    write_diagnostic_reports(diag_rows, diag_active, diag_archive, diag_manual, diag_ledger)
    write_manual_pack(manual_pack)
    write_global_reports(
        main_rows,
        audit,
        strict,
        lexical,
        diag_rows,
        diag_active,
        diag_archive,
        diag_manual,
        missing_inputs,
    )

    print(f"Wrote semantic cleanup reports to {OUT_DIR}")
    print(f"strict_main={len(strict)} lexical_stress={len(lexical)}")
    print(
        "diagnostic_active="
        f"{len(diag_active)} archive={len(diag_archive)} manual={len(diag_manual)}"
    )


if __name__ == "__main__":
    main()
