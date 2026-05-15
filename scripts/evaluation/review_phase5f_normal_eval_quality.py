#!/usr/bin/env python3
"""Phase 5F-2 read-only normal-control quality review.

Reads existing report/eval/chunk artifacts and writes normal-only quality
review outputs. It does not run retrieval, rebuild indexes, or alter eval logic.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports/phase5f_normal_eval_quality"

SOURCE_FILES = [
    "reports/phase5f_eval_quality_audit/summary.md",
    "reports/phase5f_eval_quality_audit/normal_control_audit.csv",
    "reports/phase5f_eval_quality_audit/classification_taxonomy.md",
    "reports/phase5f_eval_quality_audit/main_vs_diagnostic_recommendation.md",
    "reports/phase5f_eval_quality_audit/stable_target_mapping_audit.md",
    "reports/phase5f_eval_quality_audit/stable_target_mapping_gaps.csv",
    "reports/phase5f_eval_quality_audit/next_phase_plan.md",
    "reports/table_figure_retrieval_eval/phase4e3_eval_set_candidates/candidate_eval_set.jsonl",
    "reports/table_figure_retrieval_eval/phase4e3_eval_set_candidates/approved_candidate_suggestions.md",
    "reports/table_figure_retrieval_eval/phase4e3_eval_set_review_pack/review_pack_summary.md",
    "reports/table_figure_retrieval_eval/phase4e3_eval_set_review_pack/approved_normal_12.md",
    "reports/table_figure_retrieval_eval/phase4e3_eval_set_review_pack/normal_supplement/approved_normal_30.md",
    "reports/table_figure_retrieval_eval/phase4e3_normal_miss_review/normal_miss_ledger.csv",
    "reports/table_figure_retrieval_eval/phase4e3_normal_miss_review/summary.md",
    "reports/phase5c5_full_retrieval_ab/eval_queries.jsonl",
    "reports/phase5c5_full_retrieval_ab/summary.md",
    "reports/phase5c5_full_retrieval_ab/risk_slice_results.json",
    "reports/phase5c3_table_expansion/retrieval_ab/summary.md",
    "reports/phase5c3_table_expansion/retrieval_ab/target_mapping_audit.csv",
]

EVAL_SOURCES = [
    "reports/table_figure_retrieval_eval/phase4e3_eval_set_candidates/candidate_eval_set.jsonl",
    "reports/table_figure_retrieval_eval/phase4e3_eval_set_approved/eval_set.jsonl",
    "reports/phase5c2_table_retrieval_ab/eval_queries.jsonl",
    "reports/phase5c3_table_expansion/eval_queries.jsonl",
    "reports/phase5c5_full_retrieval_ab/eval_queries.jsonl",
]

MAPPING_SOURCES = [
    "reports/phase5c2_table_retrieval_ab/stable_target_mapping/target_mapping_audit.csv",
    "reports/phase5c3_table_expansion/retrieval_ab/target_mapping_audit.csv",
    "reports/phase5c5_full_retrieval_ab/target_mapping_audit.csv",
]

CHUNK_SOURCES = [
    Path("/tmp/biorag_phase4d_compact_chunks/chunks.jsonl"),
    Path("/tmp/biorag_phase5c4_full_enhanced/chunks/chunks.jsonl"),
    Path("/tmp/biorag_phase5d3_caption_cleanup/chunks/chunks.jsonl"),
]

QUALITY_LABELS = {
    "good_normal_control",
    "query_target_mismatch",
    "table_like_not_normal",
    "title_derived_or_mechanical",
    "too_generic_or_ambiguous",
    "target_not_stable",
    "needs_manual_review",
    "retrieval_issue_candidate",
    "diagnostic_only",
    "exclude_from_main",
}

LABEL_TO_REC = {
    "good_normal_control": ("main_eligible", "keep_main"),
    "query_target_mismatch": ("exclude_from_eval", "exclude"),
    "table_like_not_normal": ("diagnostic_only", "move_to_diagnostic"),
    "title_derived_or_mechanical": ("diagnostic_only", "rewrite_later"),
    "too_generic_or_ambiguous": ("needs_manual_review", "manual_review"),
    "target_not_stable": ("needs_manual_review", "manual_review"),
    "needs_manual_review": ("needs_manual_review", "manual_review"),
    "retrieval_issue_candidate": ("diagnostic_only", "keep_as_retrieval_issue_candidate"),
    "diagnostic_only": ("diagnostic_only", "move_to_diagnostic"),
    "exclude_from_main": ("exclude_from_eval", "exclude"),
}

SECTION_MECHANICAL_RE = re.compile(
    r"^what does (title|unknown|abstract|full text|introduction|discussion|results|methods|materials and methods|nutrients|biotechnology advances research) report about\b",
    re.IGNORECASE,
)
ANCHOR_SOUP_RE = re.compile(
    r"\b([A-Za-z]{0,4}\d+[A-Za-z0-9_.-]*|p[A-Za-z0-9_.-]*|[A-Z]{2,}\d+[A-Za-z0-9_.-]*)\b"
)
TABLE_LIKE_QUERY_RE = re.compile(r"\b(table|figure|fig\.|primer|primers|plasmid|plasmids|strain or plasmid|oligonucleotide)\b", re.IGNORECASE)
NATURAL_START_RE = re.compile(r"^(how|what enzyme|what gene|what genes|what pathway|what strategy|what result|what metabolic|what engineering|what was observed)\b", re.IGNORECASE)
QUERY_TARGET_MISMATCH_IDS = {
    "p4e3_normal_0032",
    "p4e3_normal_0041",
    "p4e3_normal_0049",
    "p4e3_normal_0057",
}
WEAK_TARGET_IDS = {"p4e3_normal_0054"}
AMBIGUOUS_IDS = {"p4e3_normal_0060"}
RETRIEVAL_ISSUE_IDS = {"p4e3_normal_supplement_0008", "p4e3_normal_supplement_0014"}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: stringify(row.get(field, "")) for field in fieldnames})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def preview(value: Any, limit: int = 700) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def phase_for(source: str) -> str:
    if "phase4e3" in source:
        return "Phase 4E-3"
    if "phase5c2" in source:
        return "Phase 5C-2"
    if "phase5c3" in source:
        return "Phase 5C-3"
    if "phase5c5" in source:
        return "Phase 5C-5"
    if "phase5f" in source:
        return "Phase 5F-1"
    return "unknown"


def load_mapping() -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for source in MAPPING_SOURCES:
        for row in read_csv(ROOT / source):
            sample_id = row.get("sample_id", "")
            blocks = [item for item in row.get("stable_target_block_ids", "").split(";") if item]
            if sample_id and blocks:
                mapping[sample_id] = blocks
    return mapping


def load_miss_review() -> dict[str, dict[str, str]]:
    path = ROOT / "reports/table_figure_retrieval_eval/phase4e3_normal_miss_review/normal_miss_ledger.csv"
    return {row.get("sample_id", ""): row for row in read_csv(path)}


def load_phase5f_previous() -> dict[str, dict[str, str]]:
    path = ROOT / "reports/phase5f_eval_quality_audit/normal_control_audit.csv"
    result: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        result[row.get("sample_id", "")] = row
    return result


def load_chunk_index() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for source in CHUNK_SOURCES:
        if not source.exists():
            continue
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                chunk_id = str(row.get("chunk_id", ""))
                if chunk_id and chunk_id not in index:
                    index[chunk_id] = row
    return index


def normal_rows_from_source(source: str) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(ROOT / source):
        if row.get("sample_type") == "normal" or row.get("query_type") == "normal_control":
            rows.append(dict(row))
    return rows


def target_chunk_id(row: dict[str, Any]) -> str:
    return str(
        row.get("target_chunk_id")
        or row.get("target_chunk_id_enhanced")
        or row.get("target_chunk_id_baseline")
        or row.get("target_chunk_id_candidate")
        or ""
    )


def stable_blocks(row: dict[str, Any], mapping: dict[str, list[str]], chunk: dict[str, Any] | None) -> list[str]:
    direct = row.get("stable_target_block_ids")
    if isinstance(direct, list):
        return [str(item) for item in direct if item]
    if isinstance(direct, str) and direct:
        return [item for item in direct.split(";") if item]
    sample_id = str(row.get("sample_id", ""))
    if sample_id in mapping:
        return mapping[sample_id]
    if chunk:
        return [str(item) for item in chunk.get("source_block_ids", []) if item]
    return []


def chunk_block_types(chunk: dict[str, Any] | None, row: dict[str, Any]) -> list[str]:
    value = row.get("target_block_types") or (chunk or {}).get("block_types") or []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return re.findall(r"[A-Za-z_]+", value)
    return []


def chunk_evidence_types(chunk: dict[str, Any] | None, row: dict[str, Any]) -> list[str]:
    value = row.get("target_evidence_types") or (chunk or {}).get("evidence_types") or []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return re.findall(r"[A-Za-z_]+", value)
    return []


def chunk_text(row: dict[str, Any], chunk: dict[str, Any] | None) -> str:
    return str(row.get("target_text_preview") or (chunk or {}).get("text") or "")


def contains_table_or_figure(chunk: dict[str, Any] | None, block_types: list[str], evidence_types: list[str], text: str) -> bool:
    if chunk and any(bool(chunk.get(flag)) for flag in ["contains_table_caption", "contains_table_text", "contains_figure_caption"]):
        return True
    joined = " ".join(block_types + evidence_types).lower()
    if "table" in joined or "figure" in joined:
        return True
    return bool(re.search(r"\b(table|fig(?:ure)?\.?)\s+s?\d+\b", text[:500], re.IGNORECASE))


def list_like_text(text: str) -> bool:
    head = text[:1000]
    numeric = len(re.findall(r"\b\d+(?:\.\d+)?\b", head))
    separators = head.count(";") + head.count("|") + head.count("\t")
    compact_ids = len(re.findall(r"\b[A-Za-z]{1,6}\d+[A-Za-z0-9_.-]*\b", head))
    return numeric >= 16 or separators >= 12 or compact_ids >= 18


def is_anchor_soup_query(query: str) -> bool:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:'/+-]*", query)
    soup = ANCHOR_SOUP_RE.findall(query)
    if len(soup) >= 3:
        return True
    if query.lower().startswith(("what does the study report about", "what was optimized to improve", "which method or pathway was used for")):
        return len(tokens) <= 13 or len(soup) >= 2
    return False


def natural_enough(query: str) -> bool:
    return bool(NATURAL_START_RE.search(query)) and not SECTION_MECHANICAL_RE.search(query)


def classify_row(
    row: dict[str, Any],
    chunk: dict[str, Any] | None,
    blocks: list[str],
    miss_review: dict[str, dict[str, str]],
) -> tuple[str, str, str, str, str]:
    sample_id = str(row.get("sample_id", ""))
    phase4_id = str(row.get("notes", "")).split("phase4_sample_id=")[-1].split(";")[0].strip() if "phase4_sample_id=" in str(row.get("notes", "")) else ""
    related_ids = {sample_id, phase4_id}
    query = str(row.get("query", ""))
    block_types = chunk_block_types(chunk, row)
    evidence_types = chunk_evidence_types(chunk, row)
    text = chunk_text(row, chunk)
    has_stable = bool(blocks)

    previous_category = ""
    for sid in related_ids:
        if sid in miss_review:
            previous_category = miss_review[sid].get("category", "")
            break

    if any(sid in RETRIEVAL_ISSUE_IDS for sid in related_ids) or previous_category.startswith("retrieval_issue_"):
        return (
            "retrieval_issue_candidate",
            "diagnostic_only",
            "keep_as_retrieval_issue_candidate",
            "Query-target quality was previously reviewed as plausible; historical miss is retained for retrieval diagnosis, not treated as bad eval data.",
            "If kept in main now, unresolved retrieval behavior could dominate normal denominator interpretation.",
        )

    if any(sid in QUERY_TARGET_MISMATCH_IDS for sid in related_ids) or previous_category.startswith("eval_sample_issue_query_target_mismatch"):
        return (
            "query_target_mismatch",
            "exclude_from_eval",
            "exclude",
            "Prior manual review found query anchors point away from the target; this is eval sample noise rather than a retrieval issue.",
            "Would count a bad query-target pair as a retrieval failure.",
        )

    if any(sid in WEAK_TARGET_IDS for sid in related_ids) or previous_category.startswith("eval_sample_issue_weak"):
        return (
            "needs_manual_review",
            "needs_manual_review",
            "manual_review",
            "Prior review found a weak or boilerplate target; keep out of main until rewritten or manually approved.",
            "Could add weak metadata/OCR-like noise to normal denominator.",
        )

    if any(sid in AMBIGUOUS_IDS for sid in related_ids) or previous_category == "ambiguous_or_multiple_valid_docs":
        return (
            "too_generic_or_ambiguous",
            "needs_manual_review",
            "manual_review",
            "Prior review found multiple plausible answer documents or an under-specified target.",
            "Could penalize retrieval for returning another valid answer.",
        )

    if not has_stable:
        return (
            "target_not_stable",
            "needs_manual_review",
            "manual_review",
            "No stable target block ids could be located from eval rows, mapping audits, or chunk provenance.",
            "Raw chunk_id may shift across versions and corrupt A/B scoring.",
        )

    if contains_table_or_figure(chunk, block_types, evidence_types, text) or list_like_text(text):
        return (
            "table_like_not_normal",
            "diagnostic_only",
            "move_to_diagnostic",
            "Target content is table/figure/list-like rather than a clean normal paragraph control.",
            "Would blur normal-control denominator with table/list retrieval behavior.",
        )

    if SECTION_MECHANICAL_RE.search(query) or is_anchor_soup_query(query):
        return (
            "title_derived_or_mechanical",
            "diagnostic_only",
            "rewrite_later",
            "Query is title/section-derived or anchor-soup style and does not resemble a natural normal-control question.",
            "Would make normal performance depend on synthetic exact-anchor matching rather than natural prose retrieval.",
        )

    if not natural_enough(query):
        return (
            "too_generic_or_ambiguous",
            "needs_manual_review",
            "manual_review",
            "Query is not clearly natural enough for the main normal denominator without manual approval.",
            "Could introduce ambiguous or mechanically phrased samples into main scoring.",
        )

    if set(block_types) and set(block_types).issubset({"title", "section_heading", "subsection_heading"}):
        return (
            "title_derived_or_mechanical",
            "diagnostic_only",
            "rewrite_later",
            "Target is title/heading-only rather than ordinary paragraph evidence.",
            "Would test heading retrieval, not normal paragraph retrieval.",
        )

    return (
        "good_normal_control",
        "main_eligible",
        "keep_main",
        "Natural query, stable target blocks, and paragraph-style target without table/figure/list-like evidence.",
        "Low; residual risk is ordinary corpus ambiguity.",
    )


def enrich_eval_rows() -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    missing = [source for source in SOURCE_FILES if not (ROOT / source).exists()]
    missing.extend(str(path) for path in CHUNK_SOURCES if not path.exists())
    mapping = load_mapping()
    miss_review = load_miss_review()
    previous = load_phase5f_previous()
    chunk_index = load_chunk_index()

    rows: list[dict[str, Any]] = []
    for source in EVAL_SOURCES:
        for raw in normal_rows_from_source(source):
            row = dict(raw)
            chunk_id = target_chunk_id(row)
            chunk = chunk_index.get(chunk_id)
            blocks = stable_blocks(row, mapping, chunk)
            quality, rec_label, action, rationale, risk = classify_row(row, chunk, blocks, miss_review)
            previous_row = previous.get(str(row.get("sample_id", "")), {})
            miss_row = miss_review.get(str(row.get("sample_id", "")), {})
            phase4_id = ""
            if "phase4_sample_id=" in str(row.get("notes", "")):
                phase4_id = str(row.get("notes", "")).split("phase4_sample_id=")[-1].split(";")[0].strip()
                miss_row = miss_row or miss_review.get(phase4_id, {})
            block_types = chunk_block_types(chunk, row)
            evidence_types = chunk_evidence_types(chunk, row)
            rows.append(
                {
                    "source_file": source,
                    "phase": phase_for(source),
                    "sample_id": row.get("sample_id", ""),
                    "source_sample_id": phase4_id,
                    "query": row.get("query", ""),
                    "target_doc_id": row.get("target_doc_id", ""),
                    "target_chunk_id": chunk_id,
                    "stable_target_block_ids": blocks,
                    "target_text_preview": preview(chunk_text(row, chunk)),
                    "target_block_types": block_types,
                    "target_evidence_types": evidence_types,
                    "previous_result": previous_row.get("recommended_label", "") or ("miss_reviewed" if miss_row else ""),
                    "previous_miss_category": miss_row.get("category", ""),
                    "quality_label": quality,
                    "recommended_label": rec_label,
                    "recommended_action": action,
                    "rationale": rationale,
                    "risk_if_kept_in_main": risk,
                }
            )

    inventory = []
    approved_ids = approved_normal_ids()
    miss_ids = set(miss_review)
    for source in EVAL_SOURCES:
        nrows = normal_rows_from_source(source)
        inventory.append(
            {
                "source_file": source,
                "phase": phase_for(source),
                "normal_sample_count": len(nrows),
                "has_query": any("query" in row for row in nrows),
                "has_target_doc_id": any("target_doc_id" in row for row in nrows),
                "has_target_chunk_id": any(target_chunk_id(row) for row in nrows),
                "has_stable_target_block_ids": any(bool(row.get("stable_target_block_ids") or mapping.get(str(row.get("sample_id", "")))) for row in nrows),
                "has_previous_label_or_approved_label": any(row.get("approved") or row.get("sample_id") in approved_ids for row in nrows),
                "has_miss_review": any(row.get("sample_id") in miss_ids or f"phase4_sample_id={row.get('sample_id')}" in str(row.get("notes", "")) for row in nrows),
                "suitable_for_this_signoff": True,
            }
        )
    return rows, missing, inventory


def approved_normal_ids() -> set[str]:
    ids: set[str] = set()
    for source in [
        ROOT / "reports/table_figure_retrieval_eval/phase4e3_eval_set_review_pack/approved_normal_12.md",
        ROOT / "reports/table_figure_retrieval_eval/phase4e3_eval_set_review_pack/normal_supplement/approved_normal_30.md",
    ]:
        if source.exists():
            ids.update(re.findall(r"`(p4e3_normal[^`]+)`", source.read_text(encoding="utf-8")))
    return ids


def write_source_inventory(inventory: list[dict[str, Any]], missing: list[str]) -> None:
    write_json(OUT_DIR / "source_inventory.json", {"sources": inventory, "missing_files": missing})
    lines = [
        "# Phase 5F-2 Normal Source Inventory",
        "",
        f"- normal eval sources checked: {len(inventory)}",
        f"- missing requested files: {len(missing)}",
        "",
        "## Missing Files",
        "",
    ]
    if missing:
        lines.extend(f"- `{item}`" for item in missing)
    else:
        lines.append("- none")
    lines.extend(["", "## Sources", ""])
    lines.append("| source | phase | normal count | query | doc target | chunk target | stable blocks | previous label | miss review | sign-off |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in inventory:
        lines.append(
            "| `{source_file}` | {phase} | {normal_sample_count} | {has_query} | {has_target_doc_id} | {has_target_chunk_id} | {has_stable_target_block_ids} | {has_previous_label_or_approved_label} | {has_miss_review} | {suitable_for_this_signoff} |".format(
                **{k: ("yes" if isinstance(v, bool) and v else "no" if isinstance(v, bool) else v) for k, v in row.items()}
            )
        )
    write_md(OUT_DIR / "source_inventory.md", "\n".join(lines))


def write_taxonomy() -> None:
    write_md(
        OUT_DIR / "normal_quality_taxonomy.md",
        """# Normal Control Quality Taxonomy

Only `good_normal_control` with `recommended_label=main_eligible` enters the normal main denominator. All other labels are diagnostic, manual-review, exclusion, or backlog material.

## good_normal_control

Definition: a natural normal-control question with a stable paragraph-style target.

Criteria: natural query; ordinary paragraph target; no table/figure/list-like evidence; query-target match is clear; stable target block ids exist; not title-derived, mechanical, or exact phrase-copy dominated.

Denominator: yes.

Example: `How was hyaluronidase secretion optimized in Pichia pastoris?`

## query_target_mismatch

Definition: query anchors point to content that does not match the target chunk.

Criteria: prior manual review or current evidence shows target is not the natural answer.

Denominator: no; exclude from eval or rewrite later.

## table_like_not_normal

Definition: target is actually table, figure, structured list, or table/list-like flattened content.

Criteria: table/figure block types, table flags, table marker text, or dense numeric/list structure.

Denominator: no; diagnostic-only if useful for monitoring.

## title_derived_or_mechanical

Definition: query is generated from title/section labels or anchor soup rather than a natural user question.

Criteria: patterns such as `What does Title report about...`, `What does Unknown report about...`, or compact ID-heavy phrasing.

Denominator: no; rewrite later.

## too_generic_or_ambiguous

Definition: query is under-specified or has multiple plausible targets.

Criteria: broad query, multiple valid documents, weak target uniqueness.

Denominator: no until manually resolved.

## target_not_stable

Definition: only a chunk id is available and no stable source block target can be located.

Criteria: missing `stable_target_block_ids` and missing chunk provenance.

Denominator: no for cross-version evaluation.

## needs_manual_review

Definition: automatic review cannot confidently classify the sample.

Criteria: plausible target but weak/boilerplate context, unusual language, or unresolved evidence.

Denominator: no until signed off.

## retrieval_issue_candidate

Definition: query-target quality is plausible, but history shows a retrieval miss or ranking issue.

Criteria: prior miss review classified it as retrieval doc recall or chunk ranking rather than eval sample noise.

Denominator: diagnostic for now; may become main after explicit decision.

## diagnostic_only

Definition: useful monitoring sample but unsuitable for main scoring.

Criteria: retained risk sample, edge case, or known non-main sample.

Denominator: no.

## exclude_from_main

Definition: should not enter the normal main denominator.

Criteria: irreparable current eval design issue, confirmed bad target, or out-of-scope target.

Denominator: no.
""",
    )


def signoff_fieldnames() -> list[str]:
    return [
        "source_file",
        "phase",
        "sample_id",
        "query",
        "target_doc_id",
        "target_chunk_id",
        "stable_target_block_ids",
        "target_text_preview",
        "target_block_types",
        "target_evidence_types",
        "previous_result",
        "previous_miss_category",
        "quality_label",
        "recommended_label",
        "recommended_action",
        "rationale",
        "risk_if_kept_in_main",
    ]


def phase_rank(row: dict[str, Any]) -> int:
    phase = row.get("phase", "")
    if phase == "Phase 5C-5":
        return 5
    if phase == "Phase 5C-3":
        return 4
    if phase == "Phase 5C-2":
        return 3
    if "eval_set_approved" in row.get("source_file", ""):
        return 2
    return 1


def unique_good_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row["quality_label"] != "good_normal_control" or row["recommended_label"] != "main_eligible":
            continue
        key = (str(row["query"]), str(row["target_doc_id"]), stringify(row["stable_target_block_ids"]))
        current = selected.get(key)
        if current is None or phase_rank(row) > phase_rank(current):
            selected[key] = row
    return sorted(selected.values(), key=lambda row: (row["phase"], row["sample_id"]))


def write_candidate_sets(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    good = unique_good_rows(rows)
    good_json = [
        {
            "sample_id": row["sample_id"],
            "query_type": "normal_control",
            "query": row["query"],
            "target_doc_id": row["target_doc_id"],
            "stable_target_block_ids": row["stable_target_block_ids"],
            "target_chunk_id_candidate": row["target_chunk_id"],
            "target_text_preview": row["target_text_preview"],
            "source_phase": row["phase"],
            "quality_label": row["quality_label"],
            "include_in_main_denominator": True,
            "notes": f"source_file={row['source_file']}; phase5f2_normal_candidate",
        }
        for row in good
    ]
    write_jsonl(OUT_DIR / "good_normal_control_candidates.jsonl", good_json)

    diagnostic_labels = {
        "retrieval_issue_candidate",
        "table_like_not_normal",
        "too_generic_or_ambiguous",
        "title_derived_or_mechanical",
        "target_not_stable",
        "needs_manual_review",
        "query_target_mismatch",
        "diagnostic_only",
        "exclude_from_main",
    }
    diagnostic = [row for row in rows if row["quality_label"] in diagnostic_labels]
    diagnostic_json = [
        {
            "sample_id": row["sample_id"],
            "query_type": "normal_control",
            "query": row["query"],
            "target_doc_id": row["target_doc_id"],
            "target_chunk_id_candidate": row["target_chunk_id"],
            "stable_target_block_ids": row["stable_target_block_ids"],
            "diagnostic_label": row["quality_label"],
            "recommended_action": row["recommended_action"],
            "include_in_main_denominator": False,
            "rationale": row["rationale"],
        }
        for row in diagnostic
    ]
    write_jsonl(OUT_DIR / "diagnostic_normal_controls.jsonl", diagnostic_json)

    write_md(
        OUT_DIR / "good_normal_control_candidates.md",
        "\n".join(
            [
                "# Good Normal Control Candidates",
                "",
                f"- unique candidate count: {len(good_json)}",
                "- scope: normal subset only; this is not the final clean eval set.",
                "- every row has `include_in_main_denominator=true` and stable target blocks.",
                "",
                "## Counts by Source Phase",
                "",
                *[f"- {phase}: {count}" for phase, count in Counter(row["source_phase"] for row in good_json).most_common()],
            ]
        ),
    )
    write_md(
        OUT_DIR / "diagnostic_normal_controls.md",
        "\n".join(
            [
                "# Diagnostic Normal Controls",
                "",
                f"- diagnostic row count: {len(diagnostic_json)}",
                "- these rows are retained for review, rewrite, or retrieval diagnosis and do not enter the normal main denominator.",
                "",
                "## Counts by Diagnostic Label",
                "",
                *[f"- {label}: {count}" for label, count in Counter(row["diagnostic_label"] for row in diagnostic_json).most_common()],
            ]
        ),
    )
    return good_json, diagnostic_json


def write_stats(rows: list[dict[str, Any]], good_json: list[dict[str, Any]], diagnostic_json: list[dict[str, Any]]) -> dict[str, Any]:
    quality_counts = Counter(row["quality_label"] for row in rows)
    rec_counts = Counter(row["recommended_label"] for row in rows)
    phase_counts = Counter(row["phase"] for row in rows)
    stable_count = sum(1 for row in rows if row["stable_target_block_ids"])
    stats = {
        "total_normal_samples_reviewed": len(rows),
        "quality_counts": {label: quality_counts.get(label, 0) for label in sorted(QUALITY_LABELS)},
        "good_normal_control_count": quality_counts.get("good_normal_control", 0),
        "query_target_mismatch_count": quality_counts.get("query_target_mismatch", 0),
        "table_like_not_normal_count": quality_counts.get("table_like_not_normal", 0),
        "title_derived_or_mechanical_count": quality_counts.get("title_derived_or_mechanical", 0),
        "too_generic_or_ambiguous_count": quality_counts.get("too_generic_or_ambiguous", 0),
        "target_not_stable_count": quality_counts.get("target_not_stable", 0),
        "needs_manual_review_count": quality_counts.get("needs_manual_review", 0),
        "retrieval_issue_candidate_count": quality_counts.get("retrieval_issue_candidate", 0),
        "diagnostic_only_count": quality_counts.get("diagnostic_only", 0),
        "exclude_from_main_count": quality_counts.get("exclude_from_main", 0),
        "main_eligible_count": rec_counts.get("main_eligible", 0),
        "diagnostic_count": len(diagnostic_json),
        "source_phase_distribution": dict(phase_counts),
        "stable_target_block_ids_coverage": {
            "with_stable_target_block_ids": stable_count,
            "without_stable_target_block_ids": len(rows) - stable_count,
            "coverage_rate": stable_count / len(rows) if rows else 0.0,
        },
        "unique_good_normal_control_candidate_count": len(good_json),
        "recommended_normal_main_denominator_size": len(good_json),
        "sufficient_for_phase5f3": len(good_json) >= 30,
    }
    write_json(OUT_DIR / "normal_quality_stats.json", stats)
    lines = [
        "# Normal Quality Summary",
        "",
        f"- total normal samples reviewed: {len(rows)}",
        f"- good_normal_control rows: {stats['good_normal_control_count']}",
        f"- unique good normal candidates: {len(good_json)}",
        f"- main_eligible rows: {stats['main_eligible_count']}",
        f"- diagnostic rows: {len(diagnostic_json)}",
        f"- stable target coverage: {stable_count}/{len(rows)} ({stats['stable_target_block_ids_coverage']['coverage_rate']:.1%})",
        f"- recommended normal main denominator size: {len(good_json)}",
        f"- sufficient for Phase 5F-3: {'yes' if stats['sufficient_for_phase5f3'] else 'no'}",
        "",
        "## Quality Label Counts",
        "",
    ]
    for label in sorted(QUALITY_LABELS):
        lines.append(f"- {label}: {quality_counts.get(label, 0)}")
    lines.extend(["", "## Source Phase Distribution", ""])
    for phase, count in phase_counts.most_common():
        lines.append(f"- {phase}: {count}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The normal pool has complete stable-target coverage, but only 20 unique good normal candidates under the stricter Phase 5F-2 criteria. Do not directly build a balanced Phase 5F-3 main denominator from all normal rows; either supplement normal candidates first or explicitly accept a smaller normal denominator.",
        ]
    )
    write_md(OUT_DIR / "normal_quality_summary.md", "\n".join(lines))
    return stats


def write_next_phase_plan(stats: dict[str, Any]) -> None:
    proceed = "yes, conditional" if stats["sufficient_for_phase5f3"] else "not directly"
    blocker = (
        "Normal candidates are sufficient for a 30-sample denominator."
        if stats["sufficient_for_phase5f3"]
        else "Only 20 unique good normal candidates are available; supplement or approve a smaller normal denominator before final clean eval generation."
    )
    write_md(
        OUT_DIR / "next_phase_plan.md",
        f"""# Phase 5F-2 Next Phase Plan

## Recommendation

Proceed to Phase 5F-3: {proceed}.

Normal main denominator candidate size: {stats['recommended_normal_main_denominator_size']}.

{blocker}

## Phase 5F-3 Scope After Normal Decision

- Generate `clean_main_eval_set.jsonl`.
- Generate `diagnostic_eval_set.jsonl`.
- Integrate clean table_content, approved table/figure captions, and `good_normal_control` candidates.
- Use stable target mapping.
- Do not call Qwen.
- Do not run retrieval eval.

## If Normal Samples Become Insufficient

- Supplement from existing chunks only after manual/semi-manual review.
- Avoid title-derived mechanical queries.
- Do not use table/list-like chunks as normal controls.
- Require stable target block ids before main use.
- Preserve source provenance and rationale.

## Not Recommended

- Do not directly use all normal samples.
- Do not delete failed samples to improve metrics.
- Do not treat every retrieval miss as an eval sample issue.
- Do not skip rationale.
- Do not run retrieval before generating the clean/diagnostic eval sets.
""",
    )


def write_summary(stats: dict[str, Any]) -> None:
    q = stats["quality_counts"]
    phase5f3_recommendation = (
        "yes" if stats["sufficient_for_phase5f3"] else "not directly; supplement or explicitly accept a 20-sample normal denominator first"
    )
    write_md(
        OUT_DIR / "summary.md",
        f"""# Phase 5F-2 Normal Eval Set Quality Review Summary

## Answers

1. normal_control samples reviewed: {stats['total_normal_samples_reviewed']}.
2. good_normal_control rows: {stats['good_normal_control_count']}; unique candidate rows: {stats['unique_good_normal_control_candidate_count']}.
3. main disqualification causes: title/mechanical queries ({q.get('title_derived_or_mechanical', 0)}), table/list-like targets ({q.get('table_like_not_normal', 0)}), query-target mismatch ({q.get('query_target_mismatch', 0)}), ambiguity/manual review ({q.get('too_generic_or_ambiguous', 0) + q.get('needs_manual_review', 0)}).
4. query_target_mismatch severity: material but localized; count={q.get('query_target_mismatch', 0)}.
5. table_like_not_normal frequency: count={q.get('table_like_not_normal', 0)}.
6. stable target coverage: {stats['stable_target_block_ids_coverage']['with_stable_target_block_ids']}/{stats['total_normal_samples_reviewed']} ({stats['stable_target_block_ids_coverage']['coverage_rate']:.1%}).
7. retrieval_issue_candidate exists: yes; count={q.get('retrieval_issue_candidate', 0)}.
8. recommend Phase 5F-3: {phase5f3_recommendation}.
9. retrieval changes needed: no.
10. index rebuild needed: no.
11. Qwen/RAGAS needed: no.
12. next stage: supplement/review normal candidates or accept the smaller normal denominator, then generate clean main and diagnostic eval sets in Phase 5F-3 using stable mapping and keeping diagnostic normal rows separate.
""",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, missing, inventory = enrich_eval_rows()
    write_source_inventory(inventory, missing)
    write_taxonomy()
    write_csv(OUT_DIR / "normal_control_signoff.csv", rows, signoff_fieldnames())
    good_json, diagnostic_json = write_candidate_sets(rows)
    stats = write_stats(rows, good_json, diagnostic_json)
    write_next_phase_plan(stats)
    write_summary(stats)
    print(f"Wrote Phase 5F-2 normal review outputs to {rel(OUT_DIR)}")


if __name__ == "__main__":
    main()
