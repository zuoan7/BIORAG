#!/usr/bin/env python3
"""Build Phase 5F clean main and diagnostic eval sets.

This script only reads existing reports/eval assets/chunks and writes reports
under reports/phase5f_clean_eval_set. It does not run retrieval, rebuild
indexes, call generation judges, or modify eval pipeline logic.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports/phase5f_clean_eval_set"

SOURCES = [
    {
        "path": "reports/phase5f_eval_quality_audit/summary.md",
        "phase": "Phase 5F-1",
        "sample_type": "summary",
        "usage": "reference-only",
    },
    {
        "path": "reports/phase5f_eval_quality_audit/classification_taxonomy.md",
        "phase": "Phase 5F-1",
        "sample_type": "taxonomy",
        "usage": "reference-only",
    },
    {
        "path": "reports/phase5f_eval_quality_audit/table_figure_probe_audit.csv",
        "phase": "Phase 5F-1",
        "sample_type": "caption probes",
        "usage": "main / diagnostic",
    },
    {
        "path": "reports/phase5f_eval_quality_audit/table_content_probe_audit.csv",
        "phase": "Phase 5F-1",
        "sample_type": "table content probes",
        "usage": "main / diagnostic",
    },
    {
        "path": "reports/phase5f_eval_quality_audit/normal_control_audit.csv",
        "phase": "Phase 5F-1",
        "sample_type": "legacy normal controls",
        "usage": "diagnostic",
    },
    {
        "path": "reports/phase5f_eval_quality_audit/stable_target_mapping_audit.md",
        "phase": "Phase 5F-1",
        "sample_type": "target mapping audit",
        "usage": "reference-only",
    },
    {
        "path": "reports/phase5f_eval_quality_audit/stable_target_mapping_gaps.csv",
        "phase": "Phase 5F-1",
        "sample_type": "target mapping gaps",
        "usage": "diagnostic",
    },
    {
        "path": "reports/phase5f_eval_quality_audit/main_vs_diagnostic_recommendation.md",
        "phase": "Phase 5F-1",
        "sample_type": "recommendation",
        "usage": "reference-only",
    },
    {
        "path": "reports/phase5f_eval_quality_audit/next_phase_plan.md",
        "phase": "Phase 5F-1",
        "sample_type": "plan",
        "usage": "reference-only",
    },
    {
        "path": "reports/phase5f_normal_eval_quality/normal_control_signoff.csv",
        "phase": "Phase 5F-2",
        "sample_type": "normal signoff",
        "usage": "diagnostic / reference-only",
    },
    {
        "path": "reports/phase5f_normal_eval_quality/good_normal_control_candidates.jsonl",
        "phase": "Phase 5F-2",
        "sample_type": "normal good candidates",
        "usage": "reference-only",
    },
    {
        "path": "reports/phase5f_normal_eval_quality/diagnostic_normal_controls.jsonl",
        "phase": "Phase 5F-2",
        "sample_type": "diagnostic normal controls",
        "usage": "diagnostic",
    },
    {
        "path": "reports/phase5f_normal_eval_quality/normal_quality_summary.md",
        "phase": "Phase 5F-2",
        "sample_type": "summary",
        "usage": "reference-only",
    },
    {
        "path": "reports/phase5f_normal_eval_quality/normal_quality_stats.json",
        "phase": "Phase 5F-2",
        "sample_type": "stats",
        "usage": "reference-only",
    },
    {
        "path": "reports/phase5f_normal_eval_quality_supplement/good_normal_control_merged.jsonl",
        "phase": "Phase 5F-2B",
        "sample_type": "merged normal controls",
        "usage": "main",
    },
    {
        "path": "reports/phase5f_normal_eval_quality_supplement/good_normal_control_merged.md",
        "phase": "Phase 5F-2B",
        "sample_type": "merged normal summary",
        "usage": "reference-only",
    },
    {
        "path": "reports/phase5f_normal_eval_quality_supplement/diagnostic_normal_supplement.jsonl",
        "phase": "Phase 5F-2B",
        "sample_type": "supplement normal diagnostics",
        "usage": "diagnostic",
    },
    {
        "path": "reports/phase5f_normal_eval_quality_supplement/supplement_normal_signoff.csv",
        "phase": "Phase 5F-2B",
        "sample_type": "supplement normal signoff",
        "usage": "reference-only",
    },
    {
        "path": "reports/phase5f_normal_eval_quality_supplement/summary.md",
        "phase": "Phase 5F-2B",
        "sample_type": "summary",
        "usage": "reference-only",
    },
    {
        "path": "reports/phase5f_normal_eval_quality_supplement/supplement_stats.json",
        "phase": "Phase 5F-2B",
        "sample_type": "stats",
        "usage": "reference-only",
    },
    {
        "path": "reports/table_figure_retrieval_eval/phase4e3_eval_set_candidates/candidate_eval_set.jsonl",
        "phase": "Phase 4E-3",
        "sample_type": "legacy candidate eval set",
        "usage": "reference-only",
    },
    {
        "path": "reports/table_figure_retrieval_eval/phase4e3_eval_set_review_pack/review_pack_summary.md",
        "phase": "Phase 4E-3",
        "sample_type": "review summary",
        "usage": "reference-only",
    },
    {
        "path": "reports/table_figure_retrieval_eval/phase4e3_eval_set_review_pack/normal_supplement/approved_normal_30.md",
        "phase": "Phase 4E-3",
        "sample_type": "normal supplement",
        "usage": "reference-only",
    },
    {
        "path": "reports/table_figure_retrieval_eval/phase4e3_normal_miss_review/normal_miss_ledger.csv",
        "phase": "Phase 4E-3",
        "sample_type": "normal miss ledger",
        "usage": "reference-only",
    },
    {
        "path": "reports/phase5c5_full_retrieval_ab/eval_queries.jsonl",
        "phase": "Phase 5C-5",
        "sample_type": "eval queries",
        "usage": "reference-only",
    },
    {
        "path": "reports/phase5c5_full_retrieval_ab/risk_slice_results.json",
        "phase": "Phase 5C-5",
        "sample_type": "risk slice",
        "usage": "diagnostic",
    },
    {
        "path": "reports/phase5c5_full_retrieval_ab/summary.md",
        "phase": "Phase 5C-5",
        "sample_type": "summary",
        "usage": "reference-only",
    },
    {
        "path": "reports/phase5d_caption_cleanup_signoff/signoff_decisions.csv",
        "phase": "Phase 5D",
        "sample_type": "caption cleanup signoff",
        "usage": "diagnostic / filter",
    },
    {
        "path": "reports/phase5d_closeout/summary.md",
        "phase": "Phase 5D",
        "sample_type": "summary",
        "usage": "reference-only",
    },
    {
        "path": "reports/phase5e_closeout/summary.md",
        "phase": "Phase 5E",
        "sample_type": "summary",
        "usage": "reference-only",
    },
    {
        "path": "/tmp/biorag_phase4d_compact_chunks/chunks.jsonl",
        "phase": "Phase 4D",
        "sample_type": "chunks",
        "usage": "validation / preview",
    },
    {
        "path": "/tmp/biorag_phase5c4_full_enhanced/chunks/chunks.jsonl",
        "phase": "Phase 5C-4",
        "sample_type": "chunks",
        "usage": "validation / preview",
    },
    {
        "path": "/tmp/biorag_phase5d3_caption_cleanup/chunks/chunks.jsonl",
        "phase": "Phase 5D-3",
        "sample_type": "chunks",
        "usage": "validation / preview",
    },
]

TABLE_FIGURE_AUDIT = ROOT / "reports/phase5f_eval_quality_audit/table_figure_probe_audit.csv"
TABLE_CONTENT_AUDIT = ROOT / "reports/phase5f_eval_quality_audit/table_content_probe_audit.csv"
NORMAL_AUDIT = ROOT / "reports/phase5f_eval_quality_audit/normal_control_audit.csv"
MAPPING_GAPS = ROOT / "reports/phase5f_eval_quality_audit/stable_target_mapping_gaps.csv"
NORMAL_MERGED = ROOT / "reports/phase5f_normal_eval_quality_supplement/good_normal_control_merged.jsonl"
DIAGNOSTIC_NORMAL = ROOT / "reports/phase5f_normal_eval_quality/diagnostic_normal_controls.jsonl"
DIAGNOSTIC_NORMAL_SUPPLEMENT = ROOT / "reports/phase5f_normal_eval_quality_supplement/diagnostic_normal_supplement.jsonl"
PHASE5D_SIGNOFF = ROOT / "reports/phase5d_caption_cleanup_signoff/signoff_decisions.csv"
RISK_SLICE_RESULTS = ROOT / "reports/phase5c5_full_retrieval_ab/risk_slice_results.json"
PHASE5C5_EVAL_QUERIES = ROOT / "reports/phase5c5_full_retrieval_ab/eval_queries.jsonl"

GENERIC_CAPTION_RE = re.compile(r"^what does (table|figure|fig\.?)\s+\w+\s+(report|show)\??$", re.IGNORECASE)
FALSE_CAPTION_RE = re.compile(r"(false/fragment|safe_to_demote|number-only|fragment caption|parser false caption)", re.IGNORECASE)
CAPTION_COPY_RE = re.compile(r"(caption-copy|copy risk|caption copy)", re.IGNORECASE)
LOW_CONF_RE = re.compile(r"(low-confidence|medium-confidence|uncertain)", re.IGNORECASE)
STRUCTURED_TABLE_RE = re.compile(r"(row/cell|cell-level|row-level|structured table)", re.IGNORECASE)
OCR_IMAGE_RE = re.compile(r"\b(ocr|image-only|vision|visual inspection)\b", re.IGNORECASE)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def abs_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT / path


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def preview(value: Any, limit: int = 500) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def split_blocks(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if not value:
        return []
    return [item for item in str(value).replace(",", ";").split(";") if item]


def load_chunk_index() -> tuple[dict[str, dict[str, Any]], list[str]]:
    chunk_paths = [
        Path("/tmp/biorag_phase4d_compact_chunks/chunks.jsonl"),
        Path("/tmp/biorag_phase5c4_full_enhanced/chunks/chunks.jsonl"),
        Path("/tmp/biorag_phase5d3_caption_cleanup/chunks/chunks.jsonl"),
    ]
    index: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for path in chunk_paths:
        if not path.exists():
            missing.append(str(path))
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                chunk_id = row.get("chunk_id")
                if chunk_id and chunk_id not in index:
                    index[chunk_id] = row
    return index, missing


def chunk_preview(chunk_index: dict[str, dict[str, Any]], chunk_id: str) -> str:
    chunk = chunk_index.get(chunk_id, {})
    return preview(chunk.get("text", ""), 500)


def target_mapping_status(sample: dict[str, Any]) -> tuple[str, str]:
    stable = split_blocks(sample.get("stable_target_block_ids"))
    caption = bool(sample.get("target_caption_block_id"))
    associated = bool(sample.get("target_associated_block_id"))
    chunk = bool(sample.get("target_chunk_id_candidate"))
    if stable:
        return "stable_target", ""
    if caption or associated:
        return "partial_stable_fields", "stable_target_block_ids missing but caption/associated target fields exist"
    if chunk:
        return "target_chunk_id_only", "target only depends on chunk id candidate"
    return "missing_target_mapping", "no stable target mapping"


def normalize_caption_query_type(row: dict[str, Any]) -> str:
    query_type = str(row.get("query_type") or "")
    caption = str(row.get("caption_text") or "")
    query = str(row.get("query") or "")
    if query_type == "doc_0367_figure5":
        return "figure_caption"
    if "figure" in query_type.lower() or caption.lower().startswith(("fig", "figure")) or "figure" in query.lower():
        return "figure_caption"
    return "caption_level_table"


def caption_ability_scope(query_type: str) -> str:
    if query_type in {"caption_level_table", "figure_caption"}:
        return "caption_retrieval"
    return "diagnostic_noise_monitoring"


def caption_diagnostic_label(row: dict[str, Any], extra_issue: str = "") -> str:
    issue = " ".join([str(row.get("detected_issue", "")), extra_issue, str(row.get("rationale", "")), str(row.get("query", ""))])
    if FALSE_CAPTION_RE.search(issue):
        return "false_caption_noise"
    if GENERIC_CAPTION_RE.search(str(row.get("query", ""))):
        return "generic_caption_query"
    if "eval_only_noise" in str(row.get("recommended_label", "")):
        return "eval_only_noise"
    if CAPTION_COPY_RE.search(issue):
        return "caption_copy_risk"
    if LOW_CONF_RE.search(issue):
        return "low_confidence_table_association"
    if "no stable target block mapping" in issue.lower() or "chunk_id_only" in issue:
        return "target_not_stable"
    if row.get("recommended_label") == "needs_manual_review":
        return "needs_manual_review"
    return "eval_only_noise" if row.get("recommended_label") == "eval_only_noise" else "needs_manual_review"


def table_content_diagnostic_label(row: dict[str, Any], extra_issue: str = "") -> str:
    issue = " ".join([str(row.get("detected_issue", "")), extra_issue, str(row.get("rationale", "")), str(row.get("query", ""))])
    if STRUCTURED_TABLE_RE.search(issue):
        return "future_structured_table"
    if OCR_IMAGE_RE.search(issue):
        return "future_ocr_or_image"
    if LOW_CONF_RE.search(issue):
        return "low_confidence_table_association"
    if "chunk_id_only" in issue or "stable" in extra_issue:
        return "target_not_stable"
    if row.get("recommended_label") == "diagnostic_only":
        return "uncertain_table_association"
    return "needs_manual_review"


def normal_diagnostic_label(row: dict[str, Any]) -> str:
    if row.get("diagnostic_label"):
        return str(row.get("diagnostic_label"))
    label = str(row.get("quality_label") or row.get("normal_quality_label") or "")
    if label == "query_target_mismatch":
        return "normal_query_target_mismatch"
    if label in {
        "table_like_not_normal",
        "title_derived_or_mechanical",
        "target_not_stable",
        "needs_manual_review",
        "retrieval_issue_candidate",
    }:
        return label
    if row.get("recommended_label") == "needs_manual_review":
        return "needs_manual_review"
    if row.get("recommended_label") == "eval_only_noise":
        return "eval_only_noise"
    return "needs_manual_review"


def make_sample(
    *,
    sample_id: str,
    query_type: str,
    query: str,
    target_doc_id: str,
    stable_target_block_ids: list[str],
    target_caption_block_id: str = "",
    target_associated_block_id: str = "",
    target_chunk_id_candidate: str = "",
    source_phase: str,
    source_file: str,
    quality_label: str,
    recommended_label: str,
    include_in_main_denominator: bool,
    diagnostic_label: str,
    expected_capability: str,
    ability_scope: str,
    target_text_preview: str,
    rationale: str,
    notes: str = "",
) -> dict[str, Any]:
    sample = {
        "sample_id": sample_id,
        "query_type": query_type,
        "query": query,
        "target_doc_id": target_doc_id,
        "stable_target_block_ids": stable_target_block_ids,
        "target_caption_block_id": target_caption_block_id,
        "target_associated_block_id": target_associated_block_id,
        "target_chunk_id_candidate": target_chunk_id_candidate,
        "source_phase": source_phase,
        "source_file": source_file,
        "quality_label": quality_label,
        "recommended_label": recommended_label,
        "include_in_main_denominator": include_in_main_denominator,
        "diagnostic_label": diagnostic_label,
        "expected_capability": expected_capability,
        "ability_scope": ability_scope,
        "target_text_preview": preview(target_text_preview),
        "rationale": rationale,
        "notes": notes,
    }
    status, issue = target_mapping_status(sample)
    sample["target_mapping_status"] = status
    if issue and notes:
        sample["notes"] = f"{notes}; {issue}"
    elif issue:
        sample["notes"] = issue
    return sample


def load_source_inventory(chunk_missing: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    inventory: list[dict[str, Any]] = []
    missing: list[str] = []
    for source in SOURCES:
        path = abs_path(source["path"])
        exists = path.exists()
        read_success = False
        fields: list[str] = []
        if exists:
            try:
                if path.suffix == ".csv":
                    rows = read_csv(path)
                    fields = list(rows[0].keys()) if rows else []
                elif path.suffix == ".jsonl":
                    rows = read_jsonl(path)
                    fields = sorted(rows[0].keys()) if rows else []
                elif path.suffix == ".json":
                    payload = read_json(path)
                    if isinstance(payload, dict):
                        fields = sorted(payload.keys())
                    elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
                        fields = sorted(payload[0].keys())
                    else:
                        fields = [type(payload).__name__]
                else:
                    text = path.read_text(encoding="utf-8")
                    fields = ["markdown_text"] if text else []
                read_success = True
            except Exception as exc:  # pragma: no cover - report-only safety
                fields = [f"read_error={exc}"]
        else:
            missing.append(str(path))
        inventory.append(
            {
                "file_path": str(path),
                "phase": source["phase"],
                "sample_type": source["sample_type"],
                "exists": exists,
                "read_success": read_success,
                "fields": fields,
                "used_for": source["usage"],
            }
        )
    for path in chunk_missing:
        if path not in missing:
            missing.append(path)
    return inventory, sorted(set(missing))


def write_source_inventory(inventory: list[dict[str, Any]], missing: list[str]) -> None:
    lines = ["# Phase 5F-3 Source Inventory", "", "## Sources"]
    for row in inventory:
        lines.extend(
            [
                "",
                f"### `{row['file_path']}`",
                f"- Phase: {row['phase']}",
                f"- Sample type: {row['sample_type']}",
                f"- Exists: {'yes' if row['exists'] else 'no'}",
                f"- Read success: {'yes' if row['read_success'] else 'no'}",
                f"- Used for: {row['used_for']}",
                f"- Fields: {', '.join(row['fields']) if row['fields'] else 'none'}",
            ]
        )
    lines.extend(["", "## Missing Files"])
    if missing:
        lines.extend(f"- `{item}`" for item in missing)
    else:
        lines.append("- none")
    write_md(OUT_DIR / "source_inventory.md", "\n".join(lines))
    write_json(OUT_DIR / "source_inventory.json", {"sources": inventory, "missing_files": missing})


def write_schema() -> None:
    write_md(
        OUT_DIR / "eval_schema.md",
        """# Phase 5F-3 Unified Eval JSONL Schema

Both `clean_main_eval_set.jsonl` and `diagnostic_eval_set.jsonl` use the same top-level fields.

| field | required | description |
|---|---:|---|
| `sample_id` | yes | Stable sample identifier within this generated dataset. |
| `query_type` | yes | One of `table_content`, `caption_level_table`, `figure_caption`, `normal_control`, or diagnostic source type. |
| `query` | yes | Retrieval query text. |
| `target_doc_id` | yes | Target document id. |
| `stable_target_block_ids` | main yes, diagnostic optional | Cross-version stable block ids. Required for every main sample. |
| `target_caption_block_id` | optional | Caption block id when available. |
| `target_associated_block_id` | optional | Associated table/text block id when available. |
| `target_chunk_id_candidate` | optional | Candidate chunk id only; not a cross-version unique target. |
| `source_phase` | yes | Provenance phase. |
| `source_file` | yes | Provenance file. |
| `quality_label` | yes | Final quality label used for set generation. |
| `recommended_label` | yes | `main_eligible`, `diagnostic_only`, `needs_manual_review`, `exclude_from_eval`, or source equivalent. |
| `include_in_main_denominator` | yes | `true` only for clean main samples. |
| `diagnostic_label` | yes | Empty for main samples; required for diagnostics. |
| `expected_capability` | yes | Plain-language expected retrieval behavior. |
| `ability_scope` | yes | One of `caption_retrieval`, `table_related_text_retrieval`, `normal_paragraph_retrieval`, `diagnostic_noise_monitoring`, `future_structured_table`, `future_ocr_or_image`. |
| `target_text_preview` | yes | Short target preview for review. |
| `rationale` | yes | Why the sample is main-eligible or diagnostic. |
| `notes` | optional | Additional provenance or mapping notes. |
| `target_mapping_status` | yes | `stable_target`, `partial_stable_fields`, `target_chunk_id_only`, or `missing_target_mapping`. |

Main eval constraints:

- `stable_target_block_ids` must be non-empty.
- `target_chunk_id_candidate` is candidate metadata only.
- `include_in_main_denominator=true`.
- `recommended_label=main_eligible`.
- No `needs_manual_review`, `eval_only_noise`, `diagnostic_only`, row/cell structured table, OCR/image, or target-chunk-only samples.

Diagnostic eval constraints:

- `include_in_main_denominator=false`.
- `diagnostic_label` must be non-empty.
- Stable target fields may be incomplete, but `target_mapping_status` must make that explicit.
""",
    )


def is_caption_main_eligible(row: dict[str, str], safe_to_demote: set[tuple[str, str]]) -> tuple[bool, str]:
    stable = split_blocks(row.get("stable_target_block_ids"))
    issue = " ".join([row.get("detected_issue", ""), row.get("rationale", ""), row.get("query", "")])
    block_key_hits = [(row.get("target_doc_id", ""), block) in safe_to_demote for block in stable]
    if row.get("recommended_label") != "main_eligible":
        return False, "not main_eligible in Phase 5F-1 audit"
    if not stable:
        return False, "missing stable_target_block_ids"
    if GENERIC_CAPTION_RE.search(row.get("query", "")):
        return False, "generic caption query"
    if FALSE_CAPTION_RE.search(issue) or any(block_key_hits):
        return False, "false/fragment caption or Phase 5D safe_to_demote"
    if CAPTION_COPY_RE.search(issue):
        return False, "caption copy risk"
    return True, ""


def build_caption_sets(chunk_index: dict[str, dict[str, Any]], safe_to_demote: set[tuple[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    table_main: list[dict[str, Any]] = []
    figure_main: list[dict[str, Any]] = []
    diagnostic: list[dict[str, Any]] = []
    for row in read_csv(TABLE_FIGURE_AUDIT):
        query_type = normalize_caption_query_type(row)
        is_main, exclusion = is_caption_main_eligible(row, safe_to_demote)
        target_preview = row.get("caption_text") or chunk_preview(chunk_index, row.get("target_chunk_id", ""))
        if is_main:
            sample = make_sample(
                sample_id=row["sample_id"],
                query_type=query_type,
                query=row["query"],
                target_doc_id=row["target_doc_id"],
                stable_target_block_ids=split_blocks(row.get("stable_target_block_ids")),
                target_chunk_id_candidate=row.get("target_chunk_id", ""),
                source_phase=row.get("phase", ""),
                source_file=row.get("source_file", ""),
                quality_label="main_eligible",
                recommended_label="main_eligible",
                include_in_main_denominator=True,
                diagnostic_label="",
                expected_capability="retrieve the relevant table or figure caption text",
                ability_scope=caption_ability_scope(query_type),
                target_text_preview=target_preview,
                rationale=row.get("rationale", "Phase 5F-1 caption probe audit marked this as main_eligible."),
                notes=f"detected_issue={row.get('detected_issue', '')}",
            )
            if query_type == "figure_caption":
                figure_main.append(sample)
            else:
                table_main.append(sample)
        else:
            diagnostic.append(
                make_sample(
                    sample_id=row["sample_id"],
                    query_type=query_type,
                    query=row["query"],
                    target_doc_id=row["target_doc_id"],
                    stable_target_block_ids=split_blocks(row.get("stable_target_block_ids")),
                    target_chunk_id_candidate=row.get("target_chunk_id", ""),
                    source_phase=row.get("phase", ""),
                    source_file=row.get("source_file", ""),
                    quality_label=row.get("recommended_label", "needs_manual_review"),
                    recommended_label="diagnostic_only"
                    if row.get("recommended_label") in {"main_eligible", "eval_only_noise"}
                    else row.get("recommended_label", "needs_manual_review"),
                    include_in_main_denominator=False,
                    diagnostic_label=caption_diagnostic_label(row, exclusion),
                    expected_capability="diagnose caption retrieval noise or mapping gaps",
                    ability_scope="diagnostic_noise_monitoring",
                    target_text_preview=target_preview,
                    rationale=row.get("rationale") or exclusion,
                    notes=f"detected_issue={row.get('detected_issue', '')}; exclusion={exclusion}",
                )
            )
    return table_main, figure_main, diagnostic


def is_table_content_main_eligible(row: dict[str, str]) -> tuple[bool, str]:
    stable = split_blocks(row.get("stable_target_block_ids"))
    issue = " ".join([row.get("detected_issue", ""), row.get("rationale", ""), row.get("query", "")])
    if row.get("recommended_label") != "main_eligible":
        return False, "not main_eligible in Phase 5F-1 audit"
    if not stable:
        return False, "missing stable_target_block_ids"
    if not (row.get("target_associated_block_id") or stable):
        return False, "missing associated target and stable blocks"
    if STRUCTURED_TABLE_RE.search(issue):
        return False, "future structured table scope"
    if OCR_IMAGE_RE.search(issue):
        return False, "future OCR/image scope"
    return True, ""


def build_table_content_sets(chunk_index: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main: list[dict[str, Any]] = []
    diagnostic: list[dict[str, Any]] = []
    for row in read_csv(TABLE_CONTENT_AUDIT):
        is_main, exclusion = is_table_content_main_eligible(row)
        target_preview = chunk_preview(chunk_index, row.get("target_chunk_id", "")) or row.get("anchor_terms", "")
        if is_main:
            main.append(
                make_sample(
                    sample_id=row["sample_id"],
                    query_type="table_content",
                    query=row["query"],
                    target_doc_id=row["target_doc_id"],
                    stable_target_block_ids=split_blocks(row.get("stable_target_block_ids")),
                    target_caption_block_id=row.get("target_caption_block_id", ""),
                    target_associated_block_id=row.get("target_associated_block_id", ""),
                    target_chunk_id_candidate=row.get("target_chunk_id", ""),
                    source_phase=row.get("phase", ""),
                    source_file=row.get("source_file", ""),
                    quality_label="main_eligible",
                    recommended_label="main_eligible",
                    include_in_main_denominator=True,
                    diagnostic_label="",
                    expected_capability="retrieve table-related text associated with a stable caption or nearby table block",
                    ability_scope="table_related_text_retrieval",
                    target_text_preview=target_preview,
                    rationale=row.get("rationale", "Phase 5F-1 table_content audit marked this as main_eligible."),
                    notes=f"anchor_terms={row.get('anchor_terms', '')}; detected_issue={row.get('detected_issue', '')}",
                )
            )
        else:
            label = table_content_diagnostic_label(row, exclusion)
            ability_scope = "future_structured_table" if label == "future_structured_table" else "future_ocr_or_image" if label == "future_ocr_or_image" else "diagnostic_noise_monitoring"
            diagnostic.append(
                make_sample(
                    sample_id=row["sample_id"],
                    query_type="table_content",
                    query=row["query"],
                    target_doc_id=row["target_doc_id"],
                    stable_target_block_ids=split_blocks(row.get("stable_target_block_ids")),
                    target_caption_block_id=row.get("target_caption_block_id", ""),
                    target_associated_block_id=row.get("target_associated_block_id", ""),
                    target_chunk_id_candidate=row.get("target_chunk_id", ""),
                    source_phase=row.get("phase", ""),
                    source_file=row.get("source_file", ""),
                    quality_label=row.get("recommended_label", "needs_manual_review"),
                    recommended_label="diagnostic_only"
                    if row.get("recommended_label") == "main_eligible"
                    else row.get("recommended_label", "needs_manual_review"),
                    include_in_main_denominator=False,
                    diagnostic_label=label,
                    expected_capability="diagnose table-related query association, mapping, or future capability boundaries",
                    ability_scope=ability_scope,
                    target_text_preview=target_preview,
                    rationale=row.get("rationale") or exclusion,
                    notes=f"anchor_terms={row.get('anchor_terms', '')}; detected_issue={row.get('detected_issue', '')}; exclusion={exclusion}",
                )
            )
    return main, diagnostic


def build_normal_main() -> list[dict[str, Any]]:
    main: list[dict[str, Any]] = []
    for row in read_jsonl(NORMAL_MERGED):
        blocks = split_blocks(row.get("stable_target_block_ids"))
        if not blocks:
            continue
        main.append(
            make_sample(
                sample_id=row.get("sample_id", ""),
                query_type="normal_control",
                query=row.get("query", ""),
                target_doc_id=row.get("target_doc_id", ""),
                stable_target_block_ids=blocks,
                target_chunk_id_candidate=row.get("target_chunk_id_candidate") or row.get("source_chunk_id", ""),
                source_phase=row.get("source_phase") or row.get("source_provenance", {}).get("source_phase", "Phase 5F-2B"),
                source_file=row.get("source_file") or row.get("source_provenance", {}).get("source_file", str(NORMAL_MERGED)),
                quality_label="good_normal_control",
                recommended_label="main_eligible",
                include_in_main_denominator=True,
                diagnostic_label="",
                expected_capability="retrieve a normal paragraph answer from stable paragraph evidence",
                ability_scope="normal_paragraph_retrieval",
                target_text_preview=row.get("target_text_preview", ""),
                rationale=row.get("rationale") or "Phase 5F-2B merged good_normal_control signoff.",
                notes=row.get("notes", ""),
            )
        )
    return main


def build_normal_diagnostics(main_sample_ids: set[str]) -> list[dict[str, Any]]:
    diagnostic: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: dict[str, Any], source_file: str, source_phase: str) -> None:
        sample_id = row.get("sample_id", "")
        if not sample_id or sample_id in seen:
            return
        seen.add(sample_id)
        blocks = split_blocks(row.get("stable_target_block_ids") or row.get("target_block_ids"))
        diagnostic.append(
            make_sample(
                sample_id=f"normal_diag_{sample_id}" if sample_id in main_sample_ids else sample_id,
                query_type=row.get("query_type", "normal_control"),
                query=row.get("query", ""),
                target_doc_id=row.get("target_doc_id", ""),
                stable_target_block_ids=blocks,
                target_chunk_id_candidate=row.get("target_chunk_id_candidate") or row.get("target_chunk_id") or row.get("source_chunk_id", ""),
                source_phase=row.get("phase") or row.get("source_phase") or source_phase,
                source_file=row.get("source_file") or source_file,
                quality_label=row.get("quality_label") or row.get("normal_quality_label") or row.get("recommended_label", "needs_manual_review"),
                recommended_label=row.get("recommended_label", "diagnostic_only"),
                include_in_main_denominator=False,
                diagnostic_label=normal_diagnostic_label(row),
                expected_capability="diagnose normal-control query quality or retrieval issue candidates",
                ability_scope="diagnostic_noise_monitoring",
                target_text_preview=row.get("target_text_preview", ""),
                rationale=row.get("rationale", "Normal control did not pass good_normal_control signoff."),
                notes=row.get("risk_if_kept_in_main") or row.get("notes", ""),
            )
        )

    for row in read_jsonl(DIAGNOSTIC_NORMAL):
        add(row, str(DIAGNOSTIC_NORMAL), "Phase 5F-2")
    for row in read_jsonl(DIAGNOSTIC_NORMAL_SUPPLEMENT):
        add(row, str(DIAGNOSTIC_NORMAL_SUPPLEMENT), "Phase 5F-2B")
    for row in read_csv(NORMAL_AUDIT):
        if row.get("recommended_label") != "main_eligible":
            add(row, str(NORMAL_AUDIT), "Phase 5F-1")
    return diagnostic


def build_mapping_gap_diagnostics() -> list[dict[str, Any]]:
    diagnostic: list[dict[str, Any]] = []
    for row in read_csv(MAPPING_GAPS):
        diagnostic.append(
            make_sample(
                sample_id=f"mapping_gap_{row.get('sample_id', '')}",
                query_type=row.get("query_type", "unknown"),
                query="",
                target_doc_id=row.get("target_doc_id", ""),
                stable_target_block_ids=[],
                target_chunk_id_candidate=row.get("target_chunk_id", ""),
                source_phase=row.get("phase", "Phase 5F-1"),
                source_file=row.get("source_file", str(MAPPING_GAPS)),
                quality_label="target_not_stable",
                recommended_label="diagnostic_only",
                include_in_main_denominator=False,
                diagnostic_label="target_not_stable",
                expected_capability="monitor target mapping gaps before cross-version scoring",
                ability_scope="diagnostic_noise_monitoring",
                target_text_preview="",
                rationale=row.get("recommended_fix", "Stable target mapping gap."),
                notes=f"gap={row.get('gap', '')}; original_sample_id={row.get('sample_id', '')}",
            )
        )
    return diagnostic


def build_risk_slice_diagnostics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in read_jsonl(PHASE5C5_EVAL_QUERIES):
        if item.get("query_type") != "risk_slice" and item.get("include_in_main_denominator") is not False:
            continue
        rows.append(
            make_sample(
                sample_id=item.get("sample_id", ""),
                query_type=item.get("query_type", "risk_slice"),
                query=item.get("query", ""),
                target_doc_id=item.get("target_doc_id", ""),
                stable_target_block_ids=split_blocks(item.get("stable_target_block_ids")),
                target_caption_block_id=item.get("target_caption_block_id", ""),
                target_associated_block_id=item.get("target_associated_block_id", ""),
                target_chunk_id_candidate=item.get("target_chunk_id_enhanced_candidate")
                or item.get("target_chunk_id_baseline_candidate")
                or item.get("target_chunk_id_candidate", ""),
                source_phase="Phase 5C-5",
                source_file=str(PHASE5C5_EVAL_QUERIES),
                quality_label="risk_slice",
                recommended_label="diagnostic_only",
                include_in_main_denominator=False,
                diagnostic_label="risk_slice",
                expected_capability="monitor known risk slice behavior outside the main denominator",
                ability_scope="diagnostic_noise_monitoring",
                target_text_preview="; ".join(item.get("anchor_terms", [])) if isinstance(item.get("anchor_terms"), list) else "",
                rationale="Phase 5C-5 risk_slice query is diagnostic-only and excluded from main denominator.",
                notes=f"risk_tags={stringify(item.get('risk_tags'))}; notes={item.get('notes', '')}",
            )
        )
    if rows:
        return rows

    payload = read_json(RISK_SLICE_RESULTS)
    if not payload:
        return []
    if isinstance(payload, list):
        iterable = payload
    elif isinstance(payload, dict):
        iterable = payload.get("records") or payload.get("risk_slice_records") or payload.get("samples") or []
    else:
        iterable = []
    for idx, item in enumerate(iterable[:200], start=1):
        if not isinstance(item, dict):
            continue
        rows.append(
            make_sample(
                sample_id=f"risk_slice_{item.get('sample_id', idx)}",
                query_type=item.get("query_type", "risk_slice"),
                query=item.get("query", ""),
                target_doc_id=item.get("target_doc_id", ""),
                stable_target_block_ids=split_blocks(item.get("stable_target_block_ids")),
                target_chunk_id_candidate=item.get("target_chunk_id_candidate") or item.get("target_chunk_id", ""),
                source_phase="Phase 5C-5",
                source_file=str(RISK_SLICE_RESULTS),
                quality_label="risk_slice",
                recommended_label="diagnostic_only",
                include_in_main_denominator=False,
                diagnostic_label="risk_slice",
                expected_capability="monitor known risk slice behavior outside the main denominator",
                ability_scope="diagnostic_noise_monitoring",
                target_text_preview=item.get("target_text_preview", ""),
                rationale="Phase 5C-5 risk slice diagnostic record.",
                notes=json.dumps({k: v for k, v in item.items() if k not in {"query", "target_text_preview"}}, ensure_ascii=False, sort_keys=True),
            )
        )
    return rows


def dedupe_main(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen_ids: set[str] = set()
    seen_keys: set[tuple[str, str, tuple[str, ...]]] = set()
    result: list[dict[str, Any]] = []
    duplicate_ids = 0
    duplicate_query_targets = 0
    for row in rows:
        sample_id = row["sample_id"]
        key = (row["query"].strip().lower(), row["target_doc_id"], tuple(split_blocks(row.get("stable_target_block_ids"))))
        if sample_id in seen_ids:
            duplicate_ids += 1
            continue
        if key in seen_keys:
            duplicate_query_targets += 1
            continue
        seen_ids.add(sample_id)
        seen_keys.add(key)
        result.append(row)
    return result, {"duplicate_sample_ids_removed": duplicate_ids, "duplicate_query_target_removed": duplicate_query_targets}


def ensure_main_constraints(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("include_in_main_denominator"):
            continue
        if row.get("recommended_label") != "main_eligible":
            continue
        if not split_blocks(row.get("stable_target_block_ids")):
            continue
        if row.get("target_mapping_status") != "stable_target":
            continue
        if row.get("diagnostic_label"):
            continue
        if row.get("ability_scope") in {"future_structured_table", "future_ocr_or_image"}:
            continue
        clean.append(row)
    return clean


def quality_ledger_rows(main_rows: list[dict[str, Any]], diagnostic_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in main_rows + diagnostic_rows:
        rows.append(
            {
                "sample_id": row.get("sample_id"),
                "query_type": row.get("query_type"),
                "source_phase": row.get("source_phase"),
                "quality_label": row.get("quality_label"),
                "recommended_label": row.get("recommended_label"),
                "include_in_main_denominator": row.get("include_in_main_denominator"),
                "diagnostic_label": row.get("diagnostic_label"),
                "target_mapping_status": row.get("target_mapping_status"),
                "rationale": row.get("rationale"),
            }
        )
    return rows


def target_mapping_rows(main_rows: list[dict[str, Any]], diagnostic_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in main_rows + diagnostic_rows:
        stable = bool(split_blocks(row.get("stable_target_block_ids")))
        caption = bool(row.get("target_caption_block_id"))
        associated = bool(row.get("target_associated_block_id"))
        chunk = bool(row.get("target_chunk_id_candidate")) and not stable and not caption and not associated
        status, issue = target_mapping_status(row)
        rows.append(
            {
                "sample_id": row.get("sample_id"),
                "query_type": row.get("query_type"),
                "stable_target_block_ids_present": stable,
                "target_caption_block_id_present": caption,
                "target_associated_block_id_present": associated,
                "target_chunk_id_only": chunk,
                "target_mapping_status": status,
                "issue": issue,
            }
        )
    return rows


def duplicate_report(main_rows: list[dict[str, Any]], diagnostic_rows: list[dict[str, Any]], dedupe_stats: dict[str, Any]) -> None:
    all_rows = main_rows + diagnostic_rows
    id_counts = Counter(row["sample_id"] for row in all_rows)
    query_counts = Counter(row["query"].strip().lower() for row in all_rows if row.get("query"))
    target_query_counts: dict[str, set[str]] = defaultdict(set)
    query_target_counts: dict[str, set[str]] = defaultdict(set)
    for row in all_rows:
        target_key = f"{row.get('target_doc_id')}:{';'.join(split_blocks(row.get('stable_target_block_ids')))}"
        target_query_counts[target_key].add(row.get("query", ""))
        query_target_counts[row.get("query", "").strip().lower()].add(target_key)
    same_target_multi_query = {key: len(values) for key, values in target_query_counts.items() if key and len(values) > 1}
    same_query_multi_target = {key: len(values) for key, values in query_target_counts.items() if key and len(values) > 1}
    copy_risk = sum(1 for row in all_rows if row.get("query") and row.get("target_text_preview") and row["query"].lower().strip("?") in row["target_text_preview"].lower())
    lines = [
        "# Phase 5F-3 Duplicate and Overlap Report",
        "",
        f"- Duplicate sample_id count in final ledger: {sum(1 for count in id_counts.values() if count > 1)}",
        f"- Duplicate query count in final ledger: {sum(1 for count in query_counts.values() if count > 1)}",
        f"- Main duplicate sample_ids removed during merge: {dedupe_stats.get('duplicate_sample_ids_removed', 0)}",
        f"- Main duplicate query+target rows removed during merge: {dedupe_stats.get('duplicate_query_target_removed', 0)}",
        f"- Same target with multiple queries: {len(same_target_multi_query)}",
        f"- Same query with multiple targets: {len(same_query_multi_target)}",
        f"- Source leakage risk: {'review duplicate queries' if same_query_multi_target else 'no obvious source leakage risk'}",
        f"- Caption/table text copy risk count: {copy_risk}",
        "",
        "## Same Query Multiple Targets",
    ]
    for query, count in sorted(same_query_multi_target.items(), key=lambda item: (-item[1], item[0]))[:25]:
        lines.append(f"- `{preview(query, 120)}`: {count} targets")
    write_md(OUT_DIR / "duplicate_overlap_report.md", "\n".join(lines))


def write_selection_summaries(
    table_content_main: list[dict[str, Any]],
    table_content_diag: list[dict[str, Any]],
    table_caption_main: list[dict[str, Any]],
    figure_caption_main: list[dict[str, Any]],
    caption_diag: list[dict[str, Any]],
    normal_main: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    clean_main: list[dict[str, Any]],
) -> None:
    write_md(
        OUT_DIR / "table_content_selection_summary.md",
        "\n".join(
            [
                "# Table Content Selection Summary",
                "",
                f"- Main candidates: {len(table_content_main)}",
                f"- Diagnostic candidates: {len(table_content_diag)}",
                "- Main ability_scope: table_related_text_retrieval",
                "- Row/cell structured table queries in main: no",
                "- OCR/image queries in main: no",
            ]
        ),
    )
    write_md(
        OUT_DIR / "caption_selection_summary.md",
        "\n".join(
            [
                "# Caption Selection Summary",
                "",
                f"- Table caption main candidates: {len(table_caption_main)}",
                f"- Figure caption main candidates: {len(figure_caption_main)}",
                f"- Caption diagnostic candidates: {len(caption_diag)}",
                "- False/fragment caption and eval_only_noise samples are diagnostic only.",
                "- `doc_0367` Figure 5 remains diagnostic because the audited records lack stable target block ids.",
            ]
        ),
    )
    write_md(
        OUT_DIR / "normal_control_selection_summary.md",
        "\n".join(
            [
                "# Normal Control Selection Summary",
                "",
                f"- Normal main candidates: {len(normal_main)}",
                f"- Stable target coverage: {sum(1 for row in normal_main if split_blocks(row.get('stable_target_block_ids')))}/{len(normal_main)}",
                "- Source: Phase 5F-2B merged good normal controls only.",
                "- retrieval_issue_candidate/title_derived/table_like/mismatch samples were not added to main.",
            ]
        ),
    )
    write_md(
        OUT_DIR / "diagnostic_set_summary.md",
        "\n".join(
            [
                "# Diagnostic Set Summary",
                "",
                f"- Diagnostic eval set total: {len(diagnostic_rows)}",
                "",
                "## Diagnostic Label Counts",
                *[f"- {label}: {count}" for label, count in sorted(Counter(row['diagnostic_label'] for row in diagnostic_rows).items())],
                "",
                "- All diagnostic samples have `include_in_main_denominator=false`.",
            ]
        ),
    )
    write_md(
        OUT_DIR / "clean_main_eval_set_summary.md",
        "\n".join(
            [
                "# Clean Main Eval Set Summary",
                "",
                f"- Clean main eval set total: {len(clean_main)}",
                "",
                "## Query Type Counts",
                *[f"- {query_type}: {count}" for query_type, count in sorted(Counter(row['query_type'] for row in clean_main).items())],
                "",
                f"- Stable target coverage: {sum(1 for row in clean_main if split_blocks(row.get('stable_target_block_ids')))}/{len(clean_main)}",
                "- target_chunk_id_candidate is retained only as candidate metadata.",
            ]
        ),
    )


def write_summary(clean_main: list[dict[str, Any]], diagnostic_rows: list[dict[str, Any]]) -> None:
    qtype_counts = Counter(row["query_type"] for row in clean_main)
    stable_count = sum(1 for row in clean_main if split_blocks(row.get("stable_target_block_ids")))
    target_chunk_only_main = sum(1 for row in clean_main if row.get("target_mapping_status") == "target_chunk_id_only")
    needs_manual_main = sum(1 for row in clean_main if row.get("recommended_label") == "needs_manual_review" or row.get("quality_label") == "needs_manual_review")
    eval_noise_main = sum(1 for row in clean_main if row.get("quality_label") == "eval_only_noise" or row.get("diagnostic_label") == "eval_only_noise")
    normal_count = qtype_counts.get("normal_control", 0)
    table_content_bad_scope = sum(1 for row in clean_main if row["query_type"] == "table_content" and row["ability_scope"] != "table_related_text_retrieval")
    future_scope_main = sum(1 for row in clean_main if row["ability_scope"] in {"future_structured_table", "future_ocr_or_image"})
    lines = [
        "# Phase 5F-3 Clean Eval Set Summary",
        "",
        f"1. clean_main_eval_set total: {len(clean_main)}",
        "2. Query type counts:",
        *[f"   - {query_type}: {count}" for query_type, count in sorted(qtype_counts.items())],
        f"3. diagnostic_eval_set total: {len(diagnostic_rows)}",
        f"4. normal_control is 30: {'yes' if normal_count == 30 else 'no'} ({normal_count})",
        f"5. Stable target coverage is 100%: {'yes' if stable_count == len(clean_main) else 'no'} ({stable_count}/{len(clean_main)})",
        f"6. target_chunk_id_only cleared from main: {'yes' if target_chunk_only_main == 0 else 'no'}",
        f"7. needs_manual_review in main: {'no' if needs_manual_main == 0 else 'yes'}",
        f"8. eval_only_noise in main: {'no' if eval_noise_main == 0 else 'yes'}",
        f"9. table_content limited to table_related_text_retrieval: {'yes' if table_content_bad_scope == 0 else 'no'}",
        "10. Contains row/cell structured table query: no",
        f"11. Contains OCR/image query: {'no' if future_scope_main == 0 else 'yes'}",
        f"12. Recommend entering Phase 5F-4: {'yes' if normal_count == 30 and stable_count == len(clean_main) and target_chunk_only_main == 0 and needs_manual_main == 0 and eval_noise_main == 0 and future_scope_main == 0 else 'no'}",
        "13. Phase 5F-4 needs index rebuild: no; prefer existing experiment indexes or lightweight retrieval sanity.",
        "14. Need Qwen/RAGAS: no",
    ]
    write_md(OUT_DIR / "summary.md", "\n".join(lines))


def write_next_phase_plan() -> None:
    write_md(
        OUT_DIR / "next_phase_plan.md",
        """# Phase 5F-4 Plan: Retrieval-only Sanity Regression

Use `reports/phase5f_clean_eval_set/clean_main_eval_set.jsonl` as the main denominator.

- Keep `diagnostic_eval_set.jsonl` separate; do not include diagnostic samples in the main denominator.
- Do not call Qwen.
- Do not run RAGAS.
- Prefer reusing Phase 5C-5 existing full baseline/enhanced indexes if they are still available.
- If indexes are unavailable, first run only BM25/in-memory or other lightweight sanity checks.
- Do not perform a large index rebuild just for this phase.
- Check whether table_content, caption-level table, figure caption, and normal_control slices remain stable under the clean set.
- Report retrieval-only sanity metrics and diagnostic slices separately.
""",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chunk_index, chunk_missing = load_chunk_index()
    inventory, missing = load_source_inventory(chunk_missing)
    write_source_inventory(inventory, missing)
    write_schema()

    safe_to_demote = {
        (row.get("doc_id", ""), row.get("block_id", ""))
        for row in read_csv(PHASE5D_SIGNOFF)
        if row.get("label") in {"safe_to_demote", "eval_only_noise"}
    }

    table_content_main, table_content_diag = build_table_content_sets(chunk_index)
    table_caption_main, figure_caption_main, caption_diag = build_caption_sets(chunk_index, safe_to_demote)
    normal_main = build_normal_main()
    normal_diag = build_normal_diagnostics({row["sample_id"] for row in normal_main})
    mapping_diag = build_mapping_gap_diagnostics()
    risk_diag = build_risk_slice_diagnostics()

    table_content_main, table_content_dedupe = dedupe_main(table_content_main)
    table_caption_main, table_caption_dedupe = dedupe_main(table_caption_main)
    figure_caption_main, figure_caption_dedupe = dedupe_main(figure_caption_main)
    normal_main, normal_dedupe = dedupe_main(normal_main)

    write_jsonl(OUT_DIR / "table_content_main_candidates.jsonl", table_content_main)
    write_jsonl(OUT_DIR / "table_content_diagnostic_candidates.jsonl", table_content_diag)
    write_jsonl(OUT_DIR / "table_caption_main_candidates.jsonl", table_caption_main)
    write_jsonl(OUT_DIR / "figure_caption_main_candidates.jsonl", figure_caption_main)
    write_jsonl(OUT_DIR / "caption_diagnostic_candidates.jsonl", caption_diag)
    write_jsonl(OUT_DIR / "normal_control_main_candidates.jsonl", normal_main)

    diagnostic_rows = table_content_diag + caption_diag + normal_diag + mapping_diag + risk_diag
    # Keep diagnostic sample ids unique without deleting source records silently.
    seen_diag: Counter[str] = Counter()
    for row in diagnostic_rows:
        seen_diag[row["sample_id"]] += 1
        if seen_diag[row["sample_id"]] > 1:
            row["sample_id"] = f"{row['sample_id']}__diagdup{seen_diag[row['sample_id']]}"

    raw_main = table_content_main + table_caption_main + figure_caption_main + normal_main
    constrained_main = ensure_main_constraints(raw_main)
    clean_main, dedupe_stats = dedupe_main(constrained_main)
    dedupe_stats = {
        "duplicate_sample_ids_removed": dedupe_stats.get("duplicate_sample_ids_removed", 0)
        + table_content_dedupe.get("duplicate_sample_ids_removed", 0)
        + table_caption_dedupe.get("duplicate_sample_ids_removed", 0)
        + figure_caption_dedupe.get("duplicate_sample_ids_removed", 0)
        + normal_dedupe.get("duplicate_sample_ids_removed", 0),
        "duplicate_query_target_removed": dedupe_stats.get("duplicate_query_target_removed", 0)
        + table_content_dedupe.get("duplicate_query_target_removed", 0)
        + table_caption_dedupe.get("duplicate_query_target_removed", 0)
        + figure_caption_dedupe.get("duplicate_query_target_removed", 0)
        + normal_dedupe.get("duplicate_query_target_removed", 0),
    }

    write_jsonl(OUT_DIR / "diagnostic_eval_set.jsonl", diagnostic_rows)
    write_jsonl(OUT_DIR / "clean_main_eval_set.jsonl", clean_main)

    write_selection_summaries(
        table_content_main,
        table_content_diag,
        table_caption_main,
        figure_caption_main,
        caption_diag,
        normal_main,
        diagnostic_rows,
        clean_main,
    )

    write_csv(
        OUT_DIR / "eval_quality_ledger.csv",
        quality_ledger_rows(clean_main, diagnostic_rows),
        [
            "sample_id",
            "query_type",
            "source_phase",
            "quality_label",
            "recommended_label",
            "include_in_main_denominator",
            "diagnostic_label",
            "target_mapping_status",
            "rationale",
        ],
    )
    write_csv(
        OUT_DIR / "target_mapping_audit.csv",
        target_mapping_rows(clean_main, diagnostic_rows),
        [
            "sample_id",
            "query_type",
            "stable_target_block_ids_present",
            "target_caption_block_id_present",
            "target_associated_block_id_present",
            "target_chunk_id_only",
            "target_mapping_status",
            "issue",
        ],
    )
    duplicate_report(clean_main, diagnostic_rows, dedupe_stats)
    write_summary(clean_main, diagnostic_rows)
    write_next_phase_plan()


if __name__ == "__main__":
    main()
