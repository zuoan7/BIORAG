#!/usr/bin/env python3
"""Phase 5F-1 read-only audit for eval/probe noise.

The script only consumes existing reports and eval files. It does not run
retrieval, rebuild indexes, call generation models, or modify eval logic.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports/phase5f_eval_quality_audit"

REQUESTED_SOURCES = [
    "reports/table_figure_retrieval_eval/phase4e1_failure_diagnosis/failure_ledger.csv",
    "reports/table_figure_retrieval_eval/phase4e1_failure_diagnosis/summary.md",
    "reports/table_figure_retrieval_eval/phase4e2_clean_eval_probes/probe_quality_report.json",
    "reports/table_figure_retrieval_eval/phase4e2_clean_eval_probes/excluded_probe_examples.md",
    "reports/table_figure_retrieval_eval/phase4e2_validation/summary.md",
    "reports/table_figure_retrieval_eval/phase4e2_validation/query_caption_overlap.json",
    "reports/table_figure_retrieval_eval/phase4e3_eval_set_candidates/candidate_quality_summary.json",
    "reports/table_figure_retrieval_eval/phase4e3_eval_set_candidates/approved_candidate_suggestions.md",
    "reports/table_figure_retrieval_eval/phase4e3_eval_set_review_pack/review_pack_summary.md",
    "reports/table_figure_retrieval_eval/phase4e3_eval_set_review_pack/normal_gap_to_fill.md",
    "reports/table_figure_retrieval_eval/phase4e3_normal_miss_review/normal_miss_ledger.csv",
    "reports/table_figure_retrieval_eval/phase4e3_normal_miss_review/summary.md",
    "reports/phase5c2_table_retrieval_ab/eval_queries.jsonl",
    "reports/phase5c2_table_retrieval_ab/probe_quality_report.json",
    "reports/phase5c2_table_retrieval_ab/stable_target_mapping/summary.md",
    "reports/phase5c2_table_retrieval_ab/stable_target_mapping/target_mapping_audit.csv",
    "reports/phase5c3_table_expansion/retrieval_ab/summary.md",
    "reports/phase5c3_table_expansion/retrieval_ab/target_mapping_audit.csv",
    "reports/phase5c5_full_retrieval_ab/eval_queries.jsonl",
    "reports/phase5c5_full_retrieval_ab/risk_slice_results.json",
    "reports/phase5c5_full_retrieval_ab/summary.md",
    "reports/phase5c8_closeout/summary.md",
    "reports/phase5c8_closeout/backlog.md",
    "reports/phase5d_caption_cleanup_audit/false_caption_candidates.csv",
    "reports/phase5d_caption_cleanup_signoff/signoff_decisions.csv",
    "reports/phase5d_caption_cleanup_signoff/summary.md",
    "reports/phase5d_closeout/summary.md",
    "reports/phase5d_closeout/backlog.md",
    "reports/phase5e_closeout/summary.md",
    "reports/phase5e_closeout/backlog.md",
    "reports/phase5e_closeout/decision_note.md",
]

EXTRA_SOURCES = [
    "reports/table_figure_retrieval_eval/phase4e3_eval_set_approved/eval_set.jsonl",
    "reports/table_figure_retrieval_eval/phase4e3_eval_set_approved/sanity_anchor.jsonl",
    "reports/table_figure_retrieval_eval/phase4e3_eval_set_candidates/candidate_eval_set.jsonl",
    "reports/phase5c3_table_expansion/eval_queries.jsonl",
    "reports/phase5c5_full_retrieval_ab/target_mapping_audit.csv",
    "reports/table_figure_retrieval_eval/phase4e2_validation/probe_denominator_audit.json",
]

GENERIC_QUERY_RE = re.compile(
    r"^\s*(what does|which|where)\s+(table|fig(?:ure)?)\s+s?\d+[a-z]?\s+(report|show|shows|contain|contains|list|lists)\??\s*$",
    re.IGNORECASE,
)
NUMBER_ONLY_CAPTION_RE = re.compile(r"^\s*(table|fig(?:ure)?)\.?\s+s?\d+[a-z]?\s*[:.]?\s*$", re.IGNORECASE)
FRAGMENT_CAPTION_RE = re.compile(
    r"^\s*(table|fig(?:ure)?)\.?\s+s?\d+[a-z]?\s*[:.]?\s*(the|a|an)?\s*[A-Z]\s*\.?\s*$",
    re.IGNORECASE,
)
TABLE_OR_FIGURE_ONLY_RE = re.compile(r"\b(table|fig(?:ure)?)\b", re.IGNORECASE)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: stringify(row.get(field, "")) for field in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def preview(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def phase_for(path: str) -> str:
    if "phase4e1" in path:
        return "Phase 4E-1"
    if "phase4e2" in path:
        return "Phase 4E-2"
    if "phase4e3" in path:
        return "Phase 4E-3"
    if "phase5c2" in path:
        return "Phase 5C-2"
    if "phase5c3" in path:
        return "Phase 5C-3"
    if "phase5c5" in path:
        return "Phase 5C-5"
    if "phase5c8" in path:
        return "Phase 5C closeout"
    if "phase5d" in path:
        return "Phase 5D"
    if "phase5e" in path:
        return "Phase 5E"
    return "unknown"


def sample_type_for(path: str) -> str:
    name = Path(path).name
    if name == "eval_queries.jsonl" or name in {"eval_set.jsonl", "sanity_anchor.jsonl", "candidate_eval_set.jsonl"}:
        return "eval queries"
    if "failure_ledger" in path or "normal_miss_ledger" in path:
        return "failure ledger"
    if "mapping" in path:
        return "stable target mapping"
    if "signoff" in path or "decisions" in path:
        return "sign-off labels"
    if "risk_slice" in path:
        return "risk slice diagnostics"
    if "probe_quality" in path or "probe_denominator" in path:
        return "probe quality summary"
    if "backlog" in path:
        return "backlog"
    if "summary" in path or "decision_note" in path:
        return "summary/decision note"
    return "review artifact"


def summarize_jsonl(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    type_counts = Counter(row.get("query_type") or row.get("sample_type") or row.get("eval_group") or "unknown" for row in rows)
    field_counts = Counter()
    for row in rows:
        field_counts.update(row.keys())
    return {"row_count": len(rows), "type_counts": dict(type_counts), "fields": sorted(field_counts)}


def summarize_csv(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    fields = list(rows[0].keys()) if rows else []
    return {"row_count": len(rows), "fields": fields}


def build_inventory() -> tuple[list[dict[str, Any]], list[str]]:
    all_sources = REQUESTED_SOURCES + EXTRA_SOURCES
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for source in all_sources:
        path = ROOT / source
        exists = path.exists()
        if not exists and source in REQUESTED_SOURCES:
            missing.append(source)
        summary: dict[str, Any] = {}
        if exists and path.suffix == ".jsonl":
            summary = summarize_jsonl(path)
        elif exists and path.suffix == ".csv":
            summary = summarize_csv(path)
        elif exists and path.suffix == ".json":
            try:
                payload = read_json(path)
                summary = {"top_level_keys": sorted(payload.keys()) if isinstance(payload, dict) else [], "json_type": type(payload).__name__}
            except json.JSONDecodeError:
                summary = {"json_error": "decode_failed"}

        lower = source.lower()
        has_eval = exists and path.suffix == ".jsonl" and (
            "eval_queries" in lower
            or lower.endswith("eval_set.jsonl")
            or lower.endswith("candidate_eval_set.jsonl")
            or lower.endswith("sanity_anchor.jsonl")
        )
        has_failure = exists and ("failure_ledger" in lower or "miss_ledger" in lower)
        has_labels = exists and any(token in lower for token in ["approved", "signoff", "decisions", "quality", "denominator", "review_pack"])
        has_mapping = exists and ("stable_target" in lower or "target_mapping" in lower or "mapping" in lower)
        is_risk_or_backlog = any(token in lower for token in ["risk_slice", "backlog", "diagnostic"])
        can_main = has_eval and not is_risk_or_backlog and "candidate" not in lower and "sanity_anchor" not in lower
        diagnostic = is_risk_or_backlog or has_failure or "sanity_anchor" in lower or "candidate" in lower

        rows.append(
            {
                "file_path": source,
                "exists": exists,
                "phase": phase_for(source),
                "sample_type": sample_type_for(source),
                "contains_eval_queries": has_eval,
                "contains_failure_ledger": has_failure,
                "contains_approved_rejected_diagnostic_label": has_labels,
                "contains_stable_target_mapping": has_mapping,
                "can_be_used_for_main_denominator": bool(can_main),
                "better_as_diagnostic_only": bool(diagnostic),
                "summary": summary,
            }
        )
    return rows, missing


def load_mapping_rows() -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for source in [
        "reports/phase5c2_table_retrieval_ab/stable_target_mapping/target_mapping_audit.csv",
        "reports/phase5c3_table_expansion/retrieval_ab/target_mapping_audit.csv",
        "reports/phase5c5_full_retrieval_ab/target_mapping_audit.csv",
    ]:
        for row in read_csv(ROOT / source):
            sample_id = row.get("sample_id", "")
            if sample_id:
                mapping[sample_id] = row
    return mapping


def load_signoff_by_block() -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row.get("doc_id", ""), row.get("block_id", "")): row
        for row in read_csv(ROOT / "reports/phase5d_caption_cleanup_signoff/signoff_decisions.csv")
    }


def load_signoff_by_caption() -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(ROOT / "reports/phase5d_caption_cleanup_signoff/signoff_decisions.csv"):
        result[(row.get("doc_id", ""), row.get("caption_text", ""))] = row
    return result


def stable_blocks(row: dict[str, Any], mapping: dict[str, dict[str, str]]) -> list[str]:
    direct = row.get("stable_target_block_ids")
    if isinstance(direct, list):
        return [str(item) for item in direct if item]
    if isinstance(direct, str) and direct:
        return [item for item in direct.split(";") if item]
    mapped = mapping.get(str(row.get("sample_id", "")), {}).get("stable_target_block_ids", "")
    return [item for item in mapped.split(";") if item]


def target_chunk(row: dict[str, Any]) -> str:
    return str(row.get("target_chunk_id") or row.get("target_chunk_id_baseline") or row.get("original_baseline_target_chunk_id") or "")


def has_stable_mapping(row: dict[str, Any], mapping: dict[str, dict[str, str]]) -> bool:
    return bool(stable_blocks(row, mapping) or row.get("target_caption_block_id") or row.get("target_associated_block_id"))


def detect_caption_probe_issue(
    row: dict[str, Any],
    mapping: dict[str, dict[str, str]],
    signoff_by_block: dict[tuple[str, str], dict[str, str]],
    signoff_by_caption: dict[tuple[str, str], dict[str, str]],
) -> tuple[str, str, str, str]:
    query = str(row.get("query", ""))
    doc_id = str(row.get("target_doc_id", ""))
    caption = str(row.get("caption_text") or row.get("target_caption") or row.get("target_preview") or row.get("target_text_preview") or "")
    caption_block_id = str(row.get("target_caption_block_id") or "")
    signoff = signoff_by_block.get((doc_id, caption_block_id)) if caption_block_id else None
    if not signoff and caption:
        signoff = signoff_by_caption.get((doc_id, caption))

    issues: list[str] = []
    if signoff:
        label = signoff.get("label", "")
        if label == "safe_to_demote":
            issues.append("target caption signed off as false/fragment safe_to_demote")
        elif label == "eval_only_noise":
            issues.append("target caption signed off as eval_only_noise")
        elif label in {"uncertain", "needs_manual_pdf_check"}:
            issues.append(f"target caption signoff requires review: {label}")
    if GENERIC_QUERY_RE.search(query):
        issues.append("generic table/figure number query")
    if NUMBER_ONLY_CAPTION_RE.search(caption) or FRAGMENT_CAPTION_RE.search(caption):
        issues.append("number-only or fragment caption target")
    if row.get("caption_copy_risk") == "high":
        issues.append("high caption-copy risk")
    if "parser_false_caption" in str(row.get("miss_category", "")) or "likely_false_caption" in str(row.get("risk_slices", "")):
        issues.append("failure ledger flags parser false caption")
    if "target_caption_too_short_or_fragment" in str(row.get("miss_category", "")):
        issues.append("failure ledger flags short/fragment target")
    if not has_stable_mapping(row, mapping):
        issues.append("no stable target block mapping")

    if not issues:
        if str(row.get("sample_id", "")).startswith("p4e3_sanity_anchor") or doc_id == "doc_0367":
            return "doc_0367 protected anchor", "diagnostic_only", "keep_as_anchor", "Known-good protected sanity anchor; keep separate from denominator."
        return "none obvious", "main_eligible", "keep_main", "Specific anchor query and no known caption-noise conflict."

    issue_text = "; ".join(issues)
    if any("safe_to_demote" in item or "eval_only_noise" in item or "generic table/figure" in item or "number-only" in item for item in issues):
        return issue_text, "eval_only_noise", "exclude", "Noisy caption/probe target should not enter the main denominator."
    if any("no stable target" in item for item in issues):
        return issue_text, "needs_manual_review", "manual_review", "Caption query may be useful, but stable target mapping must be added before main use."
    return issue_text, "needs_manual_review", "manual_review", "Requires human review before denominator use."


def phase4e2_probe_rows() -> list[dict[str, Any]]:
    path = ROOT / "reports/table_figure_retrieval_eval/phase4e2_clean_eval_probes/probe_quality_report.json"
    if not path.exists():
        return []
    payload = read_json(path)
    rows: list[dict[str, Any]] = []
    for probe_type in ["table", "figure"]:
        section = payload.get(probe_type, {})
        for idx, item in enumerate(section.get("selected_examples", []) or [], start=1):
            rows.append(
                {
                    "sample_id": f"phase4e2_{probe_type}_selected_{idx:04d}",
                    "query_type": item.get("subtype", probe_type),
                    "query": item.get("query", ""),
                    "target_doc_id": item.get("target_doc_id", ""),
                    "target_chunk_id": item.get("target_chunk_id", ""),
                    "target_preview": item.get("target_preview", ""),
                    "risk_slices": item.get("risk_slices", []),
                    "source_file": rel(path),
                    "phase": "Phase 4E-2",
                }
            )
        for idx, item in enumerate(section.get("excluded_examples", []) or [], start=1):
            rows.append(
                {
                    "sample_id": f"phase4e2_{probe_type}_excluded_{idx:04d}",
                    "query_type": item.get("subtype", probe_type),
                    "query": item.get("query", ""),
                    "target_doc_id": item.get("target_doc_id", ""),
                    "target_chunk_id": item.get("target_chunk_id", ""),
                    "target_preview": item.get("target_preview", ""),
                    "risk_slices": item.get("risk_slices", []),
                    "source_file": rel(path),
                    "phase": "Phase 4E-2",
                    "excluded_reasons": item.get("reasons", []),
                }
            )
    return rows


def collect_caption_probe_audit() -> list[dict[str, Any]]:
    mapping = load_mapping_rows()
    signoff_by_block = load_signoff_by_block()
    signoff_by_caption = load_signoff_by_caption()
    source_rows: list[tuple[str, str, dict[str, Any]]] = []

    failure_path = ROOT / "reports/table_figure_retrieval_eval/phase4e1_failure_diagnosis/failure_ledger.csv"
    for row in read_csv(failure_path):
        if row.get("probe_type") in {"table", "figure"}:
            item = dict(row)
            item["sample_id"] = row.get("probe_id", "")
            item["query_type"] = row.get("subtype", "")
            item["caption_text"] = row.get("target_preview", "")
            source_rows.append((rel(failure_path), "Phase 4E-1", item))

    for row in phase4e2_probe_rows():
        source_rows.append((str(row.pop("source_file")), str(row.pop("phase")), row))

    for source in [
        "reports/table_figure_retrieval_eval/phase4e3_eval_set_approved/eval_set.jsonl",
        "reports/table_figure_retrieval_eval/phase4e3_eval_set_approved/sanity_anchor.jsonl",
        "reports/table_figure_retrieval_eval/phase4e3_eval_set_candidates/candidate_eval_set.jsonl",
        "reports/phase5c2_table_retrieval_ab/eval_queries.jsonl",
        "reports/phase5c3_table_expansion/eval_queries.jsonl",
        "reports/phase5c5_full_retrieval_ab/eval_queries.jsonl",
    ]:
        path = ROOT / source
        for row in read_jsonl(path):
            qtype = row.get("query_type")
            stype = row.get("sample_type")
            if qtype in {"caption_level_table", "figure_caption"} or stype in {"table", "figure", "sanity_anchor"}:
                source_rows.append((source, phase_for(source), row))

    seen: set[tuple[str, str]] = set()
    output: list[dict[str, Any]] = []
    for source_file, phase, row in source_rows:
        sample_id = str(row.get("sample_id") or row.get("probe_id") or "")
        dedupe_key = (source_file, sample_id)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        issue, label, action, rationale = detect_caption_probe_issue(row, mapping, signoff_by_block, signoff_by_caption)
        output.append(
            {
                "source_file": source_file,
                "phase": phase,
                "sample_id": sample_id,
                "query_type": row.get("query_type") or row.get("sample_type") or row.get("probe_type") or "",
                "query": row.get("query", ""),
                "target_doc_id": row.get("target_doc_id", ""),
                "target_chunk_id": target_chunk(row),
                "stable_target_block_ids": stable_blocks(row, mapping),
                "caption_text": row.get("caption_text") or row.get("target_caption") or row.get("target_preview") or row.get("target_text_preview") or "",
                "detected_issue": issue,
                "recommended_label": label,
                "rationale": rationale,
                "recommended_action": action,
            }
        )
    return output


def classify_table_content(row: dict[str, Any], mapping: dict[str, dict[str, str]]) -> tuple[str, str, str]:
    notes = str(row.get("notes", "")).lower()
    query = str(row.get("query", ""))
    risk_tags = row.get("risk_tags", []) or []
    issues: list[str] = []
    if not has_stable_mapping(row, mapping):
        issues.append("missing stable target block mapping")
    if "low" in notes or "low_confidence" in risk_tags:
        issues.append("low-confidence association")
    if "medium" in notes:
        issues.append("medium-confidence association")
    if any(tag in {"truncation_risk", "doc_0377", "doc_0442"} for tag in risk_tags):
        issues.append("risk-slice/truncation monitor")
    if re.search(r"\b(row|cell|column)\b", query, re.IGNORECASE) and re.search(r"\bvalue|exact|which cell\b", query, re.IGNORECASE):
        issues.append("structured row/cell-level table query")
    if query.lower().startswith(("which table lists", "which table contains", "where are")):
        issues.append("table-text/caption-anchor phrasing")

    if any("structured" in issue for issue in issues):
        return "; ".join(issues), "backlog_eval_design", "rewrite_later"
    if any("missing stable" in issue for issue in issues):
        return "; ".join(issues), "needs_manual_review", "manual_review"
    if any("low-confidence" in issue or "risk-slice" in issue for issue in issues):
        return "; ".join(issues), "diagnostic_only", "move_to_diagnostic"
    if any("medium-confidence" in issue for issue in issues):
        return "; ".join(issues), "needs_manual_review", "manual_review"
    return "; ".join(issues) or "none obvious", "main_eligible", "keep_main"


def collect_table_content_audit() -> list[dict[str, Any]]:
    mapping = load_mapping_rows()
    rows: list[dict[str, Any]] = []
    for source in [
        "reports/phase5c2_table_retrieval_ab/eval_queries.jsonl",
        "reports/phase5c3_table_expansion/eval_queries.jsonl",
        "reports/phase5c5_full_retrieval_ab/eval_queries.jsonl",
    ]:
        for row in read_jsonl(ROOT / source):
            if row.get("query_type") != "table_content":
                continue
            issue, label, action = classify_table_content(row, mapping)
            rows.append(
                {
                    "source_file": source,
                    "phase": phase_for(source),
                    "sample_id": row.get("sample_id", ""),
                    "query": row.get("query", ""),
                    "target_doc_id": row.get("target_doc_id", ""),
                    "target_chunk_id": target_chunk(row),
                    "target_caption_block_id": row.get("target_caption_block_id", ""),
                    "target_associated_block_id": row.get("target_associated_block_id", ""),
                    "stable_target_block_ids": stable_blocks(row, mapping),
                    "anchor_terms": row.get("anchor_terms", []),
                    "detected_issue": issue,
                    "recommended_label": label,
                    "rationale": table_content_rationale(label, issue),
                    "recommended_action": action,
                }
            )
    return rows


def table_content_rationale(label: str, issue: str) -> str:
    if label == "main_eligible":
        return "Within current table-related text retrieval scope with stable target fields."
    if label == "diagnostic_only":
        return "Useful monitoring sample, but association/risk status makes it unsuitable for main denominator."
    if label == "needs_manual_review":
        return "Target/query may be useful, but confidence or mapping needs review before main use."
    if label == "backlog_eval_design":
        return "Requires future structured table eval rather than current caption/content retrieval."
    return issue


def classify_normal(row: dict[str, Any], mapping: dict[str, dict[str, str]], from_ledger: bool = False) -> tuple[str, str, str, str]:
    query = str(row.get("query", ""))
    category = str(row.get("category", ""))
    target_blocks = str(row.get("target_block_types", ""))
    issues: list[str] = []
    quality = "good_normal_control"
    label = "main_eligible"
    action = "keep_main"
    rationale = "Natural enough normal-control query with no known issue."

    if from_ledger:
        if category.startswith("eval_sample_issue_query_target_mismatch"):
            return "query_target_mismatch", category, "eval_only_noise", "exclude"
        if category.startswith("eval_sample_issue_weak"):
            return "query_target_mismatch", category, "needs_manual_review", "manual_review"
        if category == "ambiguous_or_multiple_valid_docs":
            return "ambiguous", category, "needs_manual_review", "manual_review"
        if category.startswith("retrieval_issue_"):
            return "retrieval_issue_candidate", category, "diagnostic_only", "move_to_diagnostic"

    if re.search(r"\bTitle report about\b", query, re.IGNORECASE):
        issues.append("title-derived mechanical query")
        quality = "query_target_mismatch"
    if re.search(r"\b(table|figure|fig\.|plasmids?|primers?|strains?)\b", query, re.IGNORECASE):
        issues.append("table-like/list-like anchors in normal query")
        quality = "table_like_not_normal"
    if len(re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]*", query)) <= 5:
        issues.append("too generic")
        quality = "too_generic"
    if any(token in target_blocks.lower() for token in ["table", "figure"]):
        issues.append("target block includes table/figure evidence")
        quality = "table_like_not_normal"
    if not has_stable_mapping(row, mapping):
        issues.append("missing stable target block mapping")

    if issues:
        if "missing stable target block mapping" in issues and len(issues) == 1:
            quality = "needs_manual_review"
            label = "needs_manual_review"
            action = "manual_review"
            rationale = "Normal sample needs stable target mapping before cross-version main use."
        elif quality in {"query_target_mismatch", "table_like_not_normal", "too_generic"}:
            label = "needs_manual_review"
            action = "manual_review"
            rationale = "Normal-control quality is questionable and should be reviewed before main use."
        else:
            label = "needs_manual_review"
            action = "manual_review"
            rationale = "Normal-control sample has unresolved quality concerns."
    return quality, "; ".join(issues) or "none obvious", label, action if action != "keep_main" else "keep_main"


def collect_normal_audit() -> list[dict[str, Any]]:
    mapping = load_mapping_rows()
    rows: list[dict[str, Any]] = []
    ledger_path = "reports/table_figure_retrieval_eval/phase4e3_normal_miss_review/normal_miss_ledger.csv"
    for row in read_csv(ROOT / ledger_path):
        quality, issue, label, action = classify_normal(row, mapping, from_ledger=True)
        rows.append(
            {
                "source_file": ledger_path,
                "phase": "Phase 4E-3",
                "sample_id": row.get("sample_id", ""),
                "query": row.get("query", ""),
                "target_doc_id": row.get("target_doc_id", ""),
                "target_chunk_id": row.get("target_chunk_id", ""),
                "target_block_ids": "",
                "normal_quality_label": quality,
                "detected_issue": issue,
                "recommended_label": label,
                "rationale": row.get("rationale", ""),
                "recommended_action": action,
            }
        )

    for source in [
        "reports/phase5c2_table_retrieval_ab/eval_queries.jsonl",
        "reports/phase5c3_table_expansion/eval_queries.jsonl",
        "reports/phase5c5_full_retrieval_ab/eval_queries.jsonl",
    ]:
        for row in read_jsonl(ROOT / source):
            if row.get("query_type") != "normal_control":
                continue
            quality, issue, label, action = classify_normal(row, mapping)
            rows.append(
                {
                    "source_file": source,
                    "phase": phase_for(source),
                    "sample_id": row.get("sample_id", ""),
                    "query": row.get("query", ""),
                    "target_doc_id": row.get("target_doc_id", ""),
                    "target_chunk_id": target_chunk(row),
                    "target_block_ids": stable_blocks(row, mapping),
                    "normal_quality_label": quality,
                    "detected_issue": issue,
                    "recommended_label": label,
                    "rationale": normal_rationale(quality, issue),
                    "recommended_action": action,
                }
            )
    return rows


def normal_rationale(quality: str, issue: str) -> str:
    if quality == "good_normal_control":
        return "No obvious table/figure leakage or mechanical query issue found by this audit."
    if quality == "retrieval_issue_candidate":
        return "Previously reviewed as likely retrieval behavior rather than eval sample noise."
    return f"Flagged for normal-control review: {issue}."


def write_inventory(rows: list[dict[str, Any]], missing: list[str]) -> None:
    write_json(OUT_DIR / "source_inventory.json", {"sources": rows, "missing_files": missing})
    lines = [
        "# Phase 5F Source Inventory",
        "",
        f"- sources checked: {len(rows)}",
        f"- existing sources: {sum(1 for row in rows if row['exists'])}",
        f"- missing requested sources: {len(missing)}",
        "",
        "## Missing Requested Files",
        "",
    ]
    if missing:
        lines.extend(f"- `{item}`" for item in missing)
    else:
        lines.append("- none")
    lines.extend(["", "## Inventory", ""])
    lines.append("| file | phase | type | exists | eval | ledger | labels | stable mapping | main candidate | diagnostic fit |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| `{file}` | {phase} | {typ} | {exists} | {eval} | {ledger} | {labels} | {mapping} | {main} | {diag} |".format(
                file=row["file_path"],
                phase=row["phase"],
                typ=row["sample_type"],
                exists="yes" if row["exists"] else "no",
                eval="yes" if row["contains_eval_queries"] else "no",
                ledger="yes" if row["contains_failure_ledger"] else "no",
                labels="yes" if row["contains_approved_rejected_diagnostic_label"] else "no",
                mapping="yes" if row["contains_stable_target_mapping"] else "no",
                main="yes" if row["can_be_used_for_main_denominator"] else "no",
                diag="yes" if row["better_as_diagnostic_only"] else "no",
            )
        )
    write_md(OUT_DIR / "source_inventory.md", "\n".join(lines))


def write_taxonomy() -> None:
    write_md(
        OUT_DIR / "classification_taxonomy.md",
        """# Phase 5F Eval/Probe Quality Taxonomy

## Denominator Policy

Only `main_eligible` enters the main denominator. `diagnostic_only` is reported separately. `eval_only_noise`, `needs_manual_review`, `exclude_from_eval`, and `backlog_eval_design` do not enter the main denominator.

## Labels

### main_eligible

Definition: natural enough query, answerable target, clear target document/chunk, complete stable target mapping, and within current text retrieval capability.

Decision conditions:
- target has stable block identity, such as `stable_target_block_ids`, `target_caption_block_id`, or `target_associated_block_id`;
- not a false/fragment caption;
- not generic-only Table/Figure number wording;
- not pure caption-copy without semantic anchors;
- query-target match is reasonable;
- does not require OCR, image understanding, or structured table objects.

Example: an approved table/figure caption query with discriminative biological anchors and a stable target block.

### diagnostic_only

Definition: useful for monitoring known risk, but unsuitable for the main denominator.

Decision conditions:
- risk slices, low-confidence associations, parser artifact monitors, truncation monitors, or known edge cases;
- retrieval issue candidates retained for diagnosis;
- known-good anchors such as doc_0367 Figure 5 that should stay separate from the denominator.

Example: doc_0377/doc_0442 risk samples or parser artifact monitoring samples.

### eval_only_noise

Definition: noise introduced by eval/probe construction; it should not count as retrieval failure in the main denominator.

Decision conditions:
- false/fragment caption probe;
- generic `What does Table/Figure N report/show?` query;
- caption-only copy with no semantic anchor;
- ambiguous target or non-unique Table/Figure number target;
- parser artifact generated query.

Example: `Fig. 3.` as a target caption or `What does Table 1 report?` across the whole corpus.

### needs_manual_review

Definition: may be salvageable, but requires human review before use.

Decision conditions:
- query looks plausible but target uniqueness is unclear;
- normal query-target match is unnatural;
- protected short caption is ambiguous;
- section/context needs manual judgment;
- stable target mapping is missing for cross-version use.

Example: a normal-control query built from title-like anchors with an otherwise plausible target.

### exclude_from_eval

Definition: should not be part of the current eval set.

Decision conditions:
- target cannot be located;
- query is meaningless;
- target is confirmed false caption;
- query requires a capability outside current caption/content retrieval scope.

Example: target signed off as `safe_to_demote` false caption.

### backlog_eval_design

Definition: valid future evaluation idea, but outside the current eval objective.

Decision conditions:
- row/cell-level table query;
- image/OCR query;
- structured `table_object` or `figure_object` query;
- generation-only grounding query.

Example: asking for an exact table cell value before structured table extraction exists.

## Stable Target Mapping Relationship

`chunk_id` is not a cross-version stable target. Main denominator samples should carry stable source-block identity. Recommended fields:

- `stable_target_block_ids`: ordered list of source block ids used for scoring;
- `target_caption_block_id`: caption block for table/figure caption targets;
- `target_associated_block_id`: associated table-content block for table_content queries;
- `source_block_ids`: chunk-level provenance used only as supporting context, not as the sole target identity.

Cross-version A/B must score against stable block mapping first, then map to version-specific chunks.
""",
    )


def counts_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "")) for row in rows))


def write_caption_probe_outputs(rows: list[dict[str, Any]]) -> None:
    fields = [
        "source_file",
        "phase",
        "sample_id",
        "query_type",
        "query",
        "target_doc_id",
        "target_chunk_id",
        "stable_target_block_ids",
        "caption_text",
        "detected_issue",
        "recommended_label",
        "rationale",
        "recommended_action",
    ]
    write_csv(OUT_DIR / "table_figure_probe_audit.csv", rows, fields)
    by_label = counts_by(rows, "recommended_label")
    by_action = counts_by(rows, "recommended_action")
    issue_counts = Counter()
    for row in rows:
        for issue in str(row["detected_issue"]).split("; "):
            issue_counts[issue] += 1
    lines = [
        "# Table/Figure Probe Audit",
        "",
        f"- audited rows: {len(rows)}",
        f"- labels: `{json.dumps(by_label, sort_keys=True)}`",
        f"- actions: `{json.dumps(by_action, sort_keys=True)}`",
        "",
        "## Findings",
        "",
        "- Phase 4E-1 failures are dominated by parser false captions, fragment captions, and generic Table/Figure number queries.",
        "- Phase 4E-2 clean probes are credible as caption-anchor sanity probes, but query-caption overlap remains high by construction.",
        "- Phase 4E-3 approved caption samples are much cleaner, but they still need stable block mapping before being treated as cross-version targets.",
        "- Phase 5D sign-off conflicts must win over eval membership: `safe_to_demote` and `eval_only_noise` caption targets should not enter the main denominator.",
        "- `doc_0367` Figure 5 remains a protected anchor and should stay outside the main denominator.",
        "",
        "## Top Detected Issues",
        "",
    ]
    for issue, count in issue_counts.most_common(12):
        lines.append(f"- {issue}: {count}")
    write_md(OUT_DIR / "table_figure_probe_audit.md", "\n".join(lines))


def write_table_content_outputs(rows: list[dict[str, Any]]) -> None:
    fields = [
        "source_file",
        "phase",
        "sample_id",
        "query",
        "target_doc_id",
        "target_chunk_id",
        "target_caption_block_id",
        "target_associated_block_id",
        "stable_target_block_ids",
        "anchor_terms",
        "detected_issue",
        "recommended_label",
        "rationale",
        "recommended_action",
    ]
    write_csv(OUT_DIR / "table_content_probe_audit.csv", rows, fields)
    by_label = counts_by(rows, "recommended_label")
    by_phase = counts_by(rows, "phase")
    lines = [
        "# Table Content Probe Audit",
        "",
        f"- audited rows: {len(rows)}",
        f"- by phase: `{json.dumps(by_phase, sort_keys=True)}`",
        f"- labels: `{json.dumps(by_label, sort_keys=True)}`",
        "",
        "## Conclusion",
        "",
        "The Phase 5C table_content denominator is directionally trustworthy for the intended current capability: table-related text retrieval through caption-nearby association, not structured row/cell reasoning.",
        "",
        "The main residual risk is query construction. Many queries are synthetic table-anchor prompts (`Which table lists...`, `Where are... summarized in a table?`) and can copy distinctive table text. That is acceptable for a retrieval sanity set if labelled, but should not be presented as a fully natural user benchmark.",
        "",
        "Phase 5C-2 has weaker stable-target hygiene in the eval JSONL itself, although the later mapping audit supplies the mapping. Phase 5C-3 and 5C-5 are better because stable target fields are present directly in the eval rows.",
        "",
        "Low/medium-confidence associations and risk slices should move to diagnostic-only or manual review. Row/cell-level questions belong in future structured table eval design.",
    ]
    write_md(OUT_DIR / "table_content_probe_audit.md", "\n".join(lines))


def write_normal_outputs(rows: list[dict[str, Any]]) -> None:
    fields = [
        "source_file",
        "phase",
        "sample_id",
        "query",
        "target_doc_id",
        "target_chunk_id",
        "target_block_ids",
        "normal_quality_label",
        "detected_issue",
        "recommended_label",
        "rationale",
        "recommended_action",
    ]
    write_csv(OUT_DIR / "normal_control_audit.csv", rows, fields)
    by_quality = counts_by(rows, "normal_quality_label")
    by_label = counts_by(rows, "recommended_label")
    lines = [
        "# Normal Control Audit",
        "",
        f"- audited rows: {len(rows)}",
        f"- quality labels: `{json.dumps(by_quality, sort_keys=True)}`",
        f"- recommended labels: `{json.dumps(by_label, sort_keys=True)}`",
        "",
        "## Conclusion",
        "",
        "The largest normal_control issue is eval-sample quality, not table/figure takeover. Phase 4E-3 already found that most normal misses were query-target mismatch, weak/boilerplate targets, or ambiguity rather than retrieval defects.",
        "",
        "Phase 5C normal controls still include mechanical title/anchor phrasing and some table/list-like anchors. They are adequate as non-regression smoke controls, but Phase 5F-2 should build a reviewed `good_normal_control` subset before the next main denominator.",
        "",
        "Cross-version normal controls should carry stable block ids. Phase 5C-2 eval rows still rely on chunk ids in the JSONL and therefore need mapping normalization before main use.",
    ]
    write_md(OUT_DIR / "normal_control_audit.md", "\n".join(lines))


def write_stable_mapping_outputs(inventory_rows: list[dict[str, Any]]) -> None:
    gaps: list[dict[str, Any]] = []
    for source in [
        "reports/phase5c2_table_retrieval_ab/eval_queries.jsonl",
        "reports/phase5c3_table_expansion/eval_queries.jsonl",
        "reports/phase5c5_full_retrieval_ab/eval_queries.jsonl",
        "reports/table_figure_retrieval_eval/phase4e3_eval_set_approved/eval_set.jsonl",
    ]:
        path = ROOT / source
        if not path.exists():
            continue
        for row in read_jsonl(path):
            blocks = row.get("stable_target_block_ids")
            has_direct = bool(blocks)
            has_caption_or_assoc = bool(row.get("target_caption_block_id") or row.get("target_associated_block_id"))
            relies_chunk = bool(row.get("target_chunk_id") or row.get("target_chunk_id_baseline")) and not (has_direct or has_caption_or_assoc)
            if relies_chunk or not (has_direct or has_caption_or_assoc):
                gaps.append(
                    {
                        "source_file": source,
                        "phase": phase_for(source),
                        "sample_id": row.get("sample_id", ""),
                        "query_type": row.get("query_type") or row.get("sample_type") or "",
                        "target_doc_id": row.get("target_doc_id", ""),
                        "target_chunk_id": target_chunk(row),
                        "has_stable_target_block_ids": has_direct,
                        "has_target_caption_block_id": bool(row.get("target_caption_block_id")),
                        "has_target_associated_block_id": bool(row.get("target_associated_block_id")),
                        "gap": "chunk_id_only_target" if relies_chunk else "missing_explicit_stable_target_fields",
                        "recommended_fix": "add stable block ids or map through target_mapping_audit before cross-version scoring",
                    }
                )
    fields = [
        "source_file",
        "phase",
        "sample_id",
        "query_type",
        "target_doc_id",
        "target_chunk_id",
        "has_stable_target_block_ids",
        "has_target_caption_block_id",
        "has_target_associated_block_id",
        "gap",
        "recommended_fix",
    ]
    write_csv(OUT_DIR / "stable_target_mapping_gaps.csv", gaps, fields)
    lines = [
        "# Stable Target Mapping Audit",
        "",
        "## Answers",
        "",
        "- `chunk_id` is still present in several eval files, but it must not be treated as a cross-version unique target.",
        "- Phase 5C-2M corrected the key A/B artifact with `target_mapping_audit.csv`; Phase 5C-3 and Phase 5C-5 mostly carry stable target fields directly.",
        "- Phase 4E-3 approved caption samples still rely on `target_chunk_id` and need stable source-block ids before reuse in cross-version A/B.",
        "- Phase 5C-2 `eval_queries.jsonl` does not embed `stable_target_block_ids`; it depends on the separate mapping audit.",
        f"- mapping gap rows emitted: {len(gaps)}",
        "",
        "## Required Clean Eval Fields",
        "",
        "- `sample_id`, `query_type`, `query`, `target_doc_id`",
        "- `stable_target_block_ids` for all cross-version samples",
        "- `target_caption_block_id` for table/figure caption samples",
        "- `target_associated_block_id` for table_content samples",
        "- version-specific chunk ids may be retained as derived fields, but not as the stable target identity",
        "",
        "## Normalization Rule",
        "",
        "`stable_target_block_ids` should be the scoring contract. `target_caption_block_id` and `target_associated_block_id` should be normalized into that list while remaining available as role-specific fields. `source_block_ids` in chunks are provenance, not a replacement for explicit eval target fields.",
    ]
    write_md(OUT_DIR / "stable_target_mapping_audit.md", "\n".join(lines))


def write_recommendation() -> None:
    write_md(
        OUT_DIR / "main_vs_diagnostic_recommendation.md",
        """# Main vs Diagnostic Recommendation

## Main Denominator Should Include

- clean table_content queries within current table-related text retrieval scope;
- approved table caption queries with discriminative anchors;
- approved figure caption queries with discriminative anchors;
- high-quality normal controls;
- samples with complete stable target mapping.

## Diagnostic-only Should Include

- risk_slice samples;
- low-confidence table associations;
- parser artifact monitoring samples;
- truncation risk docs;
- doc_0377 / doc_0442 risk samples;
- false_caption_noise monitoring;
- section metadata backlog samples;
- doc_0367 sanity anchor, reported separately from the main denominator.

## Exclude or Backlog Should Include

- false / fragment caption targets;
- generic Table/Figure number queries;
- caption-copy queries with no semantic anchor;
- normal query-target mismatch;
- row/cell-level queries until structured table eval exists;
- OCR/image queries until OCR/image capability exists.

## Evaluation Rules

- Main denominator only contains samples that are in capability scope, have clear targets, and have complete stable mapping.
- Diagnostic-only can preserve hard or noisy samples for monitoring.
- Failed samples must not be deleted to improve metrics; each removal/demotion needs a quality label and rationale.
- Cross-version A/B must use stable target mapping, not raw `chunk_id`.
""",
    )


def write_next_phase_plan() -> None:
    write_md(
        OUT_DIR / "next_phase_plan.md",
        """# Phase 5F Next Phase Plan

## Phase 5F-2: Normal Eval Set Quality Review

- Perform manual/semi-manual review of normal controls.
- Establish a `good_normal_control` set.
- Remove or demote table-like, list-like, ambiguous, and mechanically generated normal samples.
- Preserve `retrieval_issue_candidate` rows as diagnostic-only rather than deleting them.

Verify: reviewed normal ledger has quality labels, stable target fields, and explicit main/diagnostic decisions.

## Phase 5F-3: Clean Eval Set + Diagnostic Set Generation

- Generate `clean_main_eval_set.jsonl`.
- Generate `diagnostic_eval_set.jsonl`.
- Every sample must include `quality_label`, `recommended_label`, stable target fields, and rationale.
- Do not call Qwen.
- Do not run retrieval.

Verify: JSONL schema check plus counts by query_type and quality_label.

## Phase 5F-4: Retrieval-only Sanity Regression

- Use the clean main eval set.
- Report diagnostic set separately.
- Do not run Qwen/RAGAS/generation eval.
- Check table_content, caption_level_table, figure_caption, and normal controls for stability.

Verify: retrieval-only metrics use stable target mapping and separate main vs diagnostic denominators.
""",
    )


def write_summary(
    inventory_rows: list[dict[str, Any]],
    missing: list[str],
    caption_rows: list[dict[str, Any]],
    table_rows: list[dict[str, Any]],
    normal_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Phase 5F-1 Eval / Probe Noise Audit Summary",
        "",
        "## 1. Sources Read",
        "",
    ]
    existing = [row["file_path"] for row in inventory_rows if row["exists"]]
    lines.extend(f"- `{item}`" for item in existing)
    lines.extend(["", "## 2. Missing Files", ""])
    lines.extend(f"- `{item}`" for item in missing) if missing else lines.append("- none")
    lines.extend(
        [
            "",
            "## 3. Main Noise Types",
            "",
            "- false/fragment caption targets from parser artifacts;",
            "- generic Table/Figure number queries that cannot uniquely identify a corpus target;",
            "- caption-anchor or table-text-anchor query construction risk;",
            "- normal query-target mismatch and mechanical anchor-soup normal queries;",
            "- cross-version scoring risk where raw `chunk_id` is used instead of stable source blocks;",
            "- risk slices and eval-only noise mixed into denominator-like files.",
            "",
            "## 4. Table/Figure Probe Findings",
            "",
            f"- audited rows: {len(caption_rows)}",
            f"- recommended labels: `{json.dumps(counts_by(caption_rows, 'recommended_label'), sort_keys=True)}`",
            "- Most prominent issues are Phase 4E false/fragment caption contamination, generic table/figure queries, and missing stable mapping on older approved caption samples.",
            "- Phase 5D `safe_to_demote` and `eval_only_noise` labels should override eval membership.",
            "",
            "## 5. Table Content Probe Findings",
            "",
            f"- audited rows: {len(table_rows)}",
            f"- recommended labels: `{json.dumps(counts_by(table_rows, 'recommended_label'), sort_keys=True)}`",
            "- Overall credible for current table-related text retrieval.",
            "- Not a row/cell-level structured table benchmark.",
            "- Synthetic table-anchor phrasing should be labelled and separated from natural-question claims.",
            "",
            "## 6. Normal Control Findings",
            "",
            f"- audited rows: {len(normal_rows)}",
            f"- quality labels: `{json.dumps(counts_by(normal_rows, 'normal_quality_label'), sort_keys=True)}`",
            "- Largest problem is normal sample quality: query-target mismatch, title-derived wording, table/list-like anchors, and ambiguous targets.",
            "- Phase 4E-3 found normal misses were mostly eval-sample issues rather than retrieval defects.",
            "",
            "## 7. Stable Target Mapping Status",
            "",
            "- Stable target mapping is common after Phase 5C-2M, especially in Phase 5C-3 and Phase 5C-5.",
            "- Phase 5C-2 relies on separate mapping output rather than embedding stable ids in the eval JSONL.",
            "- Phase 4E-3 approved caption samples still need stable source-block ids before cross-version reuse.",
            "- Raw `chunk_id` must not be the stable target contract.",
            "",
            "## 8. Main Denominator Definition",
            "",
            "Main denominator should contain only in-scope, answerable, target-clear, stable-mapped samples: clean table_content, approved caption queries, approved figure queries, and reviewed high-quality normal controls.",
            "",
            "## 9. Diagnostic-only Definition",
            "",
            "Diagnostic-only should contain risk slices, low-confidence associations, parser artifact monitors, truncation samples, known protected anchors, and retrieval issue candidates that are useful to monitor but unsuitable for the main denominator.",
            "",
            "## 10. Phase 5F-2 Recommendation",
            "",
            "Proceed to Phase 5F-2. The next blocker is normal-control quality review and stable target normalization, not retrieval behavior.",
            "",
            "## 11. Retrieval Change Needed",
            "",
            "No.",
            "",
            "## 12. Index Rebuild Needed",
            "",
            "No.",
            "",
            "## 13. Qwen/RAGAS Needed",
            "",
            "No.",
            "",
            "## 14. Next Stage",
            "",
            "Run Phase 5F-2 normal eval quality review, then Phase 5F-3 clean main/diagnostic eval set generation without Qwen and without retrieval eval.",
        ]
    )
    write_md(OUT_DIR / "summary.md", "\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory_rows, missing = build_inventory()
    caption_rows = collect_caption_probe_audit()
    table_rows = collect_table_content_audit()
    normal_rows = collect_normal_audit()

    write_inventory(inventory_rows, missing)
    write_taxonomy()
    write_caption_probe_outputs(caption_rows)
    write_table_content_outputs(table_rows)
    write_normal_outputs(normal_rows)
    write_stable_mapping_outputs(inventory_rows)
    write_recommendation()
    write_next_phase_plan()
    write_summary(inventory_rows, missing, caption_rows, table_rows, normal_rows)

    print(f"Wrote Phase 5F audit outputs to {rel(OUT_DIR)}")


if __name__ == "__main__":
    main()
