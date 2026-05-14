from __future__ import annotations

import csv
import copy
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ALLOWED_TYPES = {
    "paragraph",
    "subsection",
    "subsection_heading",
    "list",
    "list_item",
    "body_text",
    "unknown",
    "text",
}

DISALLOWED_TYPES = {
    "figure_caption",
    "table_caption",
    "references",
    "metadata",
    "noise",
    "image",
    "section_heading",
    "title",
    "affiliation",
    "address",
}

CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}

TABLE_KEYWORDS_RE = re.compile(
    r"\b(?:strain|plasmid|genotype|source|primer|sequence|yield|titer|titre|"
    r"condition|medium|activity|concentration|temperature|time|sample|gene|"
    r"protein|substrate|product|host|vector|enzyme|mutation|variant|species|"
    r"isolate|accession|name|permutant|unit|kd|ra|od600|length|bp|orf)\b",
    re.I,
)

CJK_TABLE_KEYWORDS_RE = re.compile(
    r"(?:表|序号|菌株|质粒|引物|浓度|条件|产量|方式|优势|劣势|反应式|能量|消耗|"
    r"甲醇|甲烷|乙烯|乙醇|基因|培养基|活性|来源|编号)"
)

BIO_ITEM_RE = re.compile(
    r"\b(?:p[A-Z][A-Za-z0-9_-]{2,}|[A-Z]{2,}\d+[A-Za-z0-9_-]*|"
    r"[a-z]{2,4}[A-Z]\d*|Δ[A-Za-z0-9_-]+|[A-Z][a-z]+\s+[a-z]+)\b"
)

REFERENCE_ENTRY_RE = re.compile(
    r"^\s*(?:\[\d+\]|\d+\.?)\s+[A-Z][A-Za-z'`-]+,\s+(?:[A-Z]\.|et\s+al\.)",
    re.I,
)

AFFILIATION_RE = re.compile(
    r"\b(?:department|university|institute|college|school|faculty|correspondence|"
    r"e-?mail|@|copyright|all rights reserved|accepted:|received:|published:)\b",
    re.I,
)

FIGURE_SIGNAL_RE = re.compile(r"^\s*(?:fig\.?|figure|图)\b|\b(?:fig\.?|figure)\s*\d", re.I)

NORMAL_PROSE_VERBS_RE = re.compile(
    r"\b(?:showed|suggested|indicated|demonstrated|observed|investigated|"
    r"measured|analyzed|compared|constructed|generated|performed|incubated|"
    r"transformed|reported|described|revealed|confirmed)\b",
    re.I,
)

TOP_LEVEL_METADATA_KEYS = {
    "table_related",
    "table_related_type",
    "table_association_rule",
    "associated_table_caption_block_id",
    "associated_table_caption_text",
    "association_confidence",
    "table_enhancement_enabled",
    "table_enhancement_mode",
    "table_enhancement_rule_hits",
}

CAPTION_METADATA_KEYS = {
    "associated_table_like_block_ids",
    "table_enhancement_associated_block_count",
}


@dataclass
class TableEnhancementRunConfig:
    mode: str = "conservative_caption_nearby"
    window_after_caption: int = 5
    window_before_caption: int = 1
    max_associated_blocks_per_caption: int = 5
    min_confidence: str = "low"
    write_audit: bool = True
    fail_on_schema_drift: bool = True
    dry_run: bool = False


@dataclass
class TableEnhancementResult:
    input_dir: Path
    output_dir: Path
    audit_dir: Path
    total_docs: int = 0
    processed_docs: int = 0
    failed_docs: list[dict[str, str]] = field(default_factory=list)
    missing_docs: list[str] = field(default_factory=list)
    docs_with_table_caption: int = 0
    table_caption_count: int = 0
    association_count: int = 0
    confidence_counts: dict[str, int] = field(default_factory=dict)
    accepted_long_prose: int = 0
    uncertain_cases: int = 0
    rejected_nearby_blocks: int = 0
    suspicious_docs: list[str] = field(default_factory=list)
    max_associations_per_doc: int = 0
    safety_gate_passed: bool = True
    safety_warnings: list[str] = field(default_factory=list)
    safety_failures: list[str] = field(default_factory=list)
    schema_drift_count: int = 0

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "audit_dir": str(self.audit_dir),
            "total_docs": self.total_docs,
            "processed_docs": self.processed_docs,
            "failed_docs": self.failed_docs,
            "missing_docs": self.missing_docs,
            "docs_with_table_caption": self.docs_with_table_caption,
            "table_caption_count": self.table_caption_count,
            "table_related_associations": self.association_count,
            "confidence_counts": self.confidence_counts,
            "accepted_long_prose": self.accepted_long_prose,
            "uncertain_cases": self.uncertain_cases,
            "rejected_nearby_blocks": self.rejected_nearby_blocks,
            "suspicious_docs": self.suspicious_docs,
            "max_associations_per_doc": self.max_associations_per_doc,
            "schema_drift_count": self.schema_drift_count,
            "safety_gate_passed": self.safety_gate_passed,
            "safety_warnings": self.safety_warnings,
            "safety_failures": self.safety_failures,
        }


def config_from_settings(config: Any) -> TableEnhancementRunConfig:
    return TableEnhancementRunConfig(
        mode=str(getattr(config, "mode", "conservative_caption_nearby")),
        window_after_caption=int(getattr(config, "window_after_caption", 5)),
        window_before_caption=int(getattr(config, "window_before_caption", 1)),
        max_associated_blocks_per_caption=int(
            getattr(config, "max_associated_blocks_per_caption", 5)
        ),
        min_confidence=str(getattr(config, "min_confidence", "low")).lower(),
        write_audit=bool(getattr(config, "write_audit", True)),
        fail_on_schema_drift=bool(getattr(config, "fail_on_schema_drift", True)),
        dry_run=bool(getattr(config, "dry_run", False)),
    )


def derive_suffixed_path(path: str | Path, suffix: str) -> Path:
    source = Path(path)
    if source.suffix:
        return source.with_name(f"{source.stem}_{suffix}{source.suffix}")
    return source.with_name(f"{source.name}_{suffix}")


def read_selected_docs(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    doc_ids = [row.get("doc_id", "").strip() for row in rows if row.get("doc_id", "").strip()]
    return _unique(doc_ids)


def run_table_enhancement(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    audit_dir: str | Path,
    config: TableEnhancementRunConfig | Any,
    selected_doc_ids: Iterable[str] | None = None,
    process_all_docs: bool = True,
) -> TableEnhancementResult:
    run_config = config if isinstance(config, TableEnhancementRunConfig) else config_from_settings(config)
    _validate_config(run_config)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    audit_path = Path(audit_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"table enhancement input_dir does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve() and not run_config.dry_run:
        raise ValueError("table enhancement output_dir must not equal input_dir")

    json_files = sorted(input_path.glob("*.json"))
    all_doc_ids = [path.stem for path in json_files]
    selected = set(_unique(selected_doc_ids or []))
    target_doc_ids = all_doc_ids if process_all_docs or not selected else [doc_id for doc_id in all_doc_ids if doc_id in selected]
    target_set = set(target_doc_ids)

    audit_path.mkdir(parents=True, exist_ok=True)
    if not run_config.dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    result = TableEnhancementResult(
        input_dir=input_path,
        output_dir=output_path,
        audit_dir=audit_path,
        total_docs=len(all_doc_ids),
    )
    all_rows: list[dict[str, Any]] = []
    doc_rows: list[dict[str, Any]] = []
    associations_by_doc: Counter[str] = Counter()

    missing_requested = sorted(selected - set(all_doc_ids)) if selected else []
    result.missing_docs = missing_requested

    for src in json_files:
        doc_id = src.stem
        dst = output_path / src.name
        if doc_id not in target_set:
            if not run_config.dry_run:
                shutil.copy2(src, dst)
            continue
        try:
            before = json.loads(src.read_text(encoding="utf-8"))
            before_original = copy.deepcopy(before)
            enhanced, rows, stats = process_doc(before, config=run_config)
            drift_errors = validate_schema_compatibility(before_original, enhanced)
            if drift_errors:
                result.schema_drift_count += len(drift_errors)
                message = f"{doc_id}: " + "; ".join(drift_errors[:5])
                result.safety_failures.append(f"schema_drift: {message}")
                if run_config.fail_on_schema_drift:
                    raise ValueError(message)
            if not run_config.dry_run:
                dst.write_text(
                    json.dumps(enhanced, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            all_rows.extend(rows)
            accepted = stats["accepted_associations"]
            associations_by_doc[doc_id] = accepted
            doc_rows.append(stats)
            result.processed_docs += 1
        except Exception as exc:  # noqa: BLE001 - audit all per-document failures.
            result.failed_docs.append({"doc_id": doc_id, "error": str(exc)})
            if not run_config.dry_run and src.exists():
                shutil.copy2(src, dst)

    _populate_result(result, all_rows, doc_rows, associations_by_doc)
    _apply_safety_gate(result)
    if run_config.write_audit:
        write_audit_outputs(audit_path, all_rows, doc_rows, result)
    return result


def process_doc(
    data: dict[str, Any],
    *,
    config: TableEnhancementRunConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    doc_id = str(data.get("doc_id") or "")
    blocks = iter_blocks(data)
    audit_rows: list[dict[str, Any]] = []
    associated_block_ids: set[str] = set()
    captions = [
        (index, block)
        for index, block in enumerate(blocks)
        if clean_type(block) == "table_caption"
    ]

    for caption_index, caption in captions:
        caption_meta = ensure_metadata(caption)
        accepted_ids: list[str] = []
        candidate_positions = list(
            range(max(0, caption_index - config.window_before_caption), caption_index)
        ) + list(
            range(caption_index + 1, min(len(blocks), caption_index + config.window_after_caption + 1))
        )

        for block_index in candidate_positions:
            block = blocks[block_index]
            if block_index == caption_index:
                continue
            bid = block_id(block)
            status, reject_reason, confidence, hits = classify_candidate(
                block,
                caption,
                block_index - caption_index,
                nearby_short_run(blocks, caption_index, block_index),
            )
            if status == "accepted" and not _confidence_allowed(confidence, config.min_confidence):
                status = "rejected"
                reject_reason = f"below_min_confidence:{config.min_confidence}"
            if status == "accepted" and bid in associated_block_ids:
                status = "rejected"
                reject_reason = "already_associated_to_nearer_caption"
                confidence = ""
            if status == "accepted" and len(accepted_ids) >= config.max_associated_blocks_per_caption:
                status = "rejected"
                reject_reason = "max_associated_blocks_per_caption"
                confidence = ""

            audit_rows.append(association_row(
                doc_id,
                caption,
                block,
                caption_index,
                block_index,
                status,
                reject_reason,
                confidence,
                hits,
            ))

            if status != "accepted":
                continue

            associated_block_ids.add(bid)
            accepted_ids.append(bid)
            block_meta = ensure_metadata(block)
            block_meta.update({
                "table_related": True,
                "table_related_type": "table_like_paragraph",
                "table_association_rule": "caption_nearby_table_like",
                "associated_table_caption_block_id": block_id(caption),
                "associated_table_caption_text": normalize_text(caption.get("text", "")),
                "association_confidence": confidence,
                "table_enhancement_enabled": True,
                "table_enhancement_mode": config.mode,
                "table_enhancement_rule_hits": hits,
                # Transitional compatibility with Phase 5C pilot consumers.
                "phase5c1_pilot": True,
                "phase5c1_rule_hits": hits,
            })

        caption_meta["associated_table_like_block_ids"] = accepted_ids
        caption_meta["table_enhancement_associated_block_count"] = len(accepted_ids)
        if accepted_ids:
            caption_meta["table_enhancement_enabled"] = True
            caption_meta["table_enhancement_mode"] = config.mode

    stats = {
        "doc_id": doc_id,
        "table_caption_count": len(captions),
        "accepted_associations": len(associated_block_ids),
        "rejected_nearby_blocks": sum(1 for row in audit_rows if row["accepted_or_rejected"] == "rejected"),
        "uncertain_cases": sum(1 for row in audit_rows if row["accepted_or_rejected"] == "uncertain"),
        "accepted_long_prose": sum(1 for row in audit_rows if _row_is_accepted_long_prose(row)),
    }
    return data, audit_rows, stats


def iter_blocks(data: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for page in data.get("pages", []) or []:
        if not isinstance(page, dict):
            continue
        for block in page.get("blocks", []) or []:
            if isinstance(block, dict):
                blocks.append(block)
    return blocks


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def preview(text: Any, limit: int = 220) -> str:
    return normalize_text(text)[:limit]


def block_id(block: dict[str, Any]) -> str:
    metadata = block.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    value = block.get("block_id") or block.get("id") or metadata.get("source_block_id")
    return str(value) if value is not None else ""


def page_value(block: dict[str, Any]) -> int | None:
    value = block.get("page") or block.get("page_number")
    if value is None:
        metadata = block.get("metadata", {}) or {}
        if isinstance(metadata, dict):
            value = metadata.get("page") or metadata.get("page_number")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clean_type(block: dict[str, Any]) -> str:
    return str(block.get("type") or "unknown")


def section_key(block: dict[str, Any]) -> str:
    section_path = block.get("section_path") or []
    if isinstance(section_path, list):
        return " > ".join(str(item) for item in section_path)
    return str(section_path)


def ensure_metadata(block: dict[str, Any]) -> dict[str, Any]:
    metadata = block.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        block["metadata"] = metadata
    return metadata


def looks_like_normal_prose(text: str) -> bool:
    words = text.split()
    sentence_boundaries = len(re.findall(r"[.!?]\s+[A-Z]", text))
    if len(words) >= 20 and sentence_boundaries >= 1:
        return True
    if len(words) >= 25 and NORMAL_PROSE_VERBS_RE.search(text):
        return True
    if len(words) >= 30 and sentence_boundaries >= 1:
        return True
    if len(words) >= 45 and sentence_boundaries >= 1 and NORMAL_PROSE_VERBS_RE.search(text):
        return True
    if len(words) >= 70 and sentence_boundaries >= 1:
        return True
    return False


def table_like_rule_hits(text: str, offset: int, nearby_short: bool) -> list[str]:
    hits: list[str] = []
    words = text.split()
    numeric_tokens = re.findall(
        r"(?<![A-Za-z])\d+(?:\.\d+)?(?:%|°C|℃|g|mg|mL|ml|L|h|min|bp|kb|MJ/kg)?",
        text,
        re.I,
    )
    separators = text.count("\t") + text.count("|")
    multi_spaces = len(re.findall(r"\S\s{2,}\S", text))
    semicolon_cells = text.count(";")
    comma_cells = text.count(",")
    bio_items = BIO_ITEM_RE.findall(text)

    if len(numeric_tokens) >= 3:
        hits.append("multiple_numeric_values")
    if len(numeric_tokens) >= 2 and re.search(r"%|°C|℃|mg|mL|ml|bp|kb|MJ/kg|OD600", text, re.I):
        hits.append("numeric_units")
    if TABLE_KEYWORDS_RE.search(text):
        hits.append("table_column_keyword")
    if CJK_TABLE_KEYWORDS_RE.search(text):
        hits.append("cjk_table_keyword")
    if separators >= 1 or multi_spaces >= 2 or semicolon_cells >= 3:
        hits.append("cell_separator_pattern")
    if comma_cells >= 4 and (len(numeric_tokens) >= 2 or TABLE_KEYWORDS_RE.search(text)):
        hits.append("comma_separated_cells")
    if len(bio_items) >= 3:
        hits.append("multiple_bio_items")
    if len(words) <= 12 and (TABLE_KEYWORDS_RE.search(text) or CJK_TABLE_KEYWORDS_RE.search(text)):
        hits.append("short_column_or_row_block")
    if nearby_short and len(words) <= 16:
        hits.append("caption_nearby_short_block_run")
    if offset > 0:
        hits.append("after_caption_window")
    elif offset < 0:
        hits.append("before_caption_window")
    return hits


def classify_candidate(
    block: dict[str, Any],
    caption: dict[str, Any],
    offset: int,
    nearby_short: bool,
) -> tuple[str, str, str, list[str]]:
    btype = clean_type(block)
    text = normalize_text(block.get("text", ""))
    caption_page = page_value(caption)
    block_page = page_value(block)
    page_distance = abs(block_page - caption_page) if block_page is not None and caption_page is not None else 0

    if btype in DISALLOWED_TYPES:
        return "rejected", "not_candidate_block_type", "", []
    if btype not in ALLOWED_TYPES:
        return "rejected", "unsupported_block_type", "", []
    if btype == "subsection_heading" and re.match(r"^\s*(?:#+\s*)?\d+(?:\.\d+)+\.?\s+\S+", text):
        return "rejected", "section_heading_shape", "", []
    if not text:
        return "rejected", "empty_text", "", []
    if page_distance > 1:
        return "rejected", "page_distance_gt_1", "", []
    if REFERENCE_ENTRY_RE.search(text):
        return "rejected", "reference_entry", "", []
    if AFFILIATION_RE.search(text):
        return "rejected", "metadata_or_affiliation_signal", "", []
    if FIGURE_SIGNAL_RE.search(text) and "table" not in text.lower() and "表" not in text:
        return "rejected", "figure_signal", "", []
    if section_key(block).lower().startswith(("references", "bibliography")):
        return "rejected", "references_section", "", []
    if len(text) < 4:
        return "rejected", "too_short", "", []

    hits = table_like_rule_hits(text, offset, nearby_short)
    support_hits = [
        hit for hit in hits
        if hit not in {"after_caption_window", "before_caption_window", "caption_nearby_short_block_run"}
    ]
    core_hits = [
        hit for hit in support_hits
        if hit in {
            "multiple_numeric_values",
            "numeric_units",
            "cell_separator_pattern",
            "comma_separated_cells",
            "short_column_or_row_block",
        }
    ]

    if looks_like_normal_prose(text):
        return "rejected", "normal_prose_shape", "", hits
    if offset < 0 and not (
        "cell_separator_pattern" in core_hits
        or ("multiple_numeric_values" in core_hits and len(text.split()) <= 20)
    ):
        return "rejected", "before_caption_not_strong_row_shape", "", hits
    if not support_hits:
        return "rejected", "no_table_like_signal", "", hits
    if not core_hits:
        return "uncertain", "support_only_without_core_table_shape", "low", hits

    if len(core_hits) >= 3 or (
        "cell_separator_pattern" in core_hits and "multiple_numeric_values" in core_hits
    ):
        return "accepted", "", "high", hits
    if len(core_hits) >= 2 or (
        "short_column_or_row_block" in core_hits and "caption_nearby_short_block_run" in hits
    ):
        return "accepted", "", "medium", hits
    if (
        "multiple_numeric_values" in core_hits
        and len(text) <= 160
        and not re.search(r"[.!?]\s+[A-Z]", text)
    ):
        return "accepted", "", "low", hits
    return "uncertain", "weak_table_like_signal", "low", hits


def nearby_short_run(blocks: list[dict[str, Any]], caption_index: int, index: int) -> bool:
    start = max(caption_index + 1, index - 1)
    end = min(len(blocks), index + 2)
    short_count = 0
    for pos in range(start, end):
        if pos == caption_index:
            continue
        text = normalize_text(blocks[pos].get("text", ""))
        if text and len(text.split()) <= 16 and clean_type(blocks[pos]) in ALLOWED_TYPES:
            short_count += 1
    return short_count >= 2


def association_row(
    doc_id: str,
    caption: dict[str, Any],
    block: dict[str, Any],
    caption_index: int,
    block_index: int,
    status: str,
    reject_reason: str,
    confidence: str,
    rule_hits: list[str],
) -> dict[str, Any]:
    caption_page = page_value(caption)
    block_page = page_value(block)
    page_distance = (
        abs(block_page - caption_page)
        if block_page is not None and caption_page is not None
        else ""
    )
    return {
        "doc_id": doc_id,
        "table_caption_block_id": block_id(caption),
        "associated_block_id": block_id(block),
        "associated_block_type": clean_type(block),
        "association_confidence": confidence,
        "rule_hits": ";".join(rule_hits),
        "associated_text_preview": preview(block.get("text", "")),
        "caption_text_preview": preview(caption.get("text", "")),
        "page_distance": page_distance,
        "block_distance": abs(block_index - caption_index),
        "accepted_or_rejected": status,
        "reject_reason": reject_reason,
    }


def validate_schema_compatibility(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(before.keys()) != set(after.keys()):
        errors.append("top_level_keys_changed")
    before_blocks = iter_blocks(before)
    after_blocks = iter_blocks(after)
    if len(before_blocks) != len(after_blocks):
        errors.append("block_count_changed")
        return errors
    for index, (old, new) in enumerate(zip(before_blocks, after_blocks)):
        for key in ("type", "text", "block_id", "id", "page", "page_number", "section_path"):
            if old.get(key) != new.get(key):
                errors.append(f"block_{index}_{key}_changed")
                break
    return errors


def write_audit_outputs(
    audit_dir: Path,
    association_rows: list[dict[str, Any]],
    doc_rows: list[dict[str, Any]],
    result: TableEnhancementResult,
) -> None:
    write_csv(audit_dir / "association_audit.csv", association_rows, ASSOCIATION_FIELDNAMES)
    write_csv(audit_dir / "doc_level_stats.csv", doc_rows, DOC_FIELDNAMES)
    (audit_dir / "association_summary.json").write_text(
        json.dumps(result.to_summary_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (audit_dir / "summary.md").write_text(render_summary_md(result), encoding="utf-8")
    (audit_dir / "false_positive_review.md").write_text(
        render_false_positive_review(association_rows, result),
        encoding="utf-8",
    )


ASSOCIATION_FIELDNAMES = [
    "doc_id",
    "table_caption_block_id",
    "associated_block_id",
    "associated_block_type",
    "association_confidence",
    "rule_hits",
    "associated_text_preview",
    "caption_text_preview",
    "page_distance",
    "block_distance",
    "accepted_or_rejected",
    "reject_reason",
]

DOC_FIELDNAMES = [
    "doc_id",
    "table_caption_count",
    "accepted_associations",
    "rejected_nearby_blocks",
    "uncertain_cases",
    "accepted_long_prose",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def render_summary_md(result: TableEnhancementResult) -> str:
    summary = result.to_summary_dict()
    lines = [
        "# Table Enhancement Summary",
        "",
        f"- total_docs: {summary['total_docs']}",
        f"- processed_docs: {summary['processed_docs']}",
        f"- failed_docs: {len(result.failed_docs)}",
        f"- docs_with_table_caption: {summary['docs_with_table_caption']}",
        f"- table_caption_count: {summary['table_caption_count']}",
        f"- table_related associations: {summary['table_related_associations']}",
        f"- high confidence: {result.confidence_counts.get('high', 0)}",
        f"- medium confidence: {result.confidence_counts.get('medium', 0)}",
        f"- low confidence: {result.confidence_counts.get('low', 0)}",
        f"- accepted_long_prose: {summary['accepted_long_prose']}",
        f"- uncertain cases: {summary['uncertain_cases']}",
        f"- rejected nearby blocks: {summary['rejected_nearby_blocks']}",
        f"- suspicious docs: {len(result.suspicious_docs)}",
        f"- max_associations_per_doc: {summary['max_associations_per_doc']}",
        f"- safety_gate_passed: {str(result.safety_gate_passed).lower()}",
        "",
        "## Safety Warnings",
        "",
    ]
    lines.extend([f"- {item}" for item in result.safety_warnings] or ["- none"])
    lines.extend(["", "## Safety Failures", ""])
    lines.extend([f"- {item}" for item in result.safety_failures] or ["- none"])
    lines.append("")
    return "\n".join(lines)


def render_false_positive_review(
    rows: list[dict[str, Any]],
    result: TableEnhancementResult,
) -> str:
    lines = ["# False Positive Review", ""]
    accepted_low = [
        row for row in rows
        if row.get("accepted_or_rejected") == "accepted"
        and row.get("association_confidence") == "low"
    ]
    uncertain = [row for row in rows if row.get("accepted_or_rejected") == "uncertain"]
    suspicious_doc_set = set(result.suspicious_docs)
    suspicious = [row for row in rows if row.get("doc_id") in suspicious_doc_set]
    sections = [
        ("Accepted Low Confidence", accepted_low[:40]),
        ("Uncertain Cases", uncertain[:40]),
        ("Suspicious Docs", suspicious[:40]),
    ]
    for title, section_rows in sections:
        lines.extend([f"## {title}", ""])
        if not section_rows:
            lines.extend(["- none", ""])
            continue
        for row in section_rows:
            reason = row.get("reject_reason") or row.get("association_confidence") or "accepted"
            lines.extend([
                f"- `{row.get('doc_id', '')}` caption `{row.get('table_caption_block_id', '')}` -> block `{row.get('associated_block_id', '')}`",
                f"  decision: {row.get('accepted_or_rejected', '')} / {reason}",
                f"  hits: {row.get('rule_hits', '')}",
                f"  block: {row.get('associated_text_preview', '')}",
            ])
        lines.append("")
    return "\n".join(lines)


def make_run_id(prefix: str = "table_enhancement") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def _populate_result(
    result: TableEnhancementResult,
    rows: list[dict[str, Any]],
    doc_rows: list[dict[str, Any]],
    associations_by_doc: Counter[str],
) -> None:
    status_counts = Counter(row["accepted_or_rejected"] for row in rows)
    confidence_counts = Counter(
        row["association_confidence"]
        for row in rows
        if row["accepted_or_rejected"] == "accepted" and row.get("association_confidence")
    )
    result.association_count = status_counts.get("accepted", 0)
    result.confidence_counts = {key: confidence_counts.get(key, 0) for key in ("high", "medium", "low")}
    result.docs_with_table_caption = sum(1 for row in doc_rows if int(row.get("table_caption_count") or 0) > 0)
    result.table_caption_count = sum(int(row.get("table_caption_count") or 0) for row in doc_rows)
    result.accepted_long_prose = sum(int(row.get("accepted_long_prose") or 0) for row in doc_rows)
    result.uncertain_cases = status_counts.get("uncertain", 0)
    result.rejected_nearby_blocks = status_counts.get("rejected", 0)
    result.max_associations_per_doc = max(associations_by_doc.values(), default=0)
    if result.association_count:
        concentrated = [
            doc_id for doc_id, count in associations_by_doc.items()
            if count >= 20 or count / result.association_count >= 0.5
        ]
        result.suspicious_docs = sorted(concentrated)


def _apply_safety_gate(result: TableEnhancementResult) -> None:
    if result.accepted_long_prose > 0:
        result.safety_warnings.append(f"accepted_long_prose={result.accepted_long_prose}")
    if result.suspicious_docs:
        result.safety_warnings.append(
            "association concentration: " + ", ".join(result.suspicious_docs[:20])
        )
    if result.failed_docs:
        result.safety_warnings.append(f"failed_docs={len(result.failed_docs)}")
    if result.schema_drift_count:
        result.safety_failures.append(f"schema_drift_count={result.schema_drift_count}")
    result.safety_gate_passed = not result.safety_failures


def _row_is_accepted_long_prose(row: dict[str, Any]) -> bool:
    if row.get("accepted_or_rejected") != "accepted":
        return False
    text = str(row.get("associated_text_preview", ""))
    return looks_like_normal_prose(text)


def _confidence_allowed(confidence: str, min_confidence: str) -> bool:
    return CONFIDENCE_ORDER.get(confidence, -1) >= CONFIDENCE_ORDER.get(min_confidence, 0)


def _validate_config(config: TableEnhancementRunConfig) -> None:
    if config.mode != "conservative_caption_nearby":
        raise ValueError(f"unsupported table enhancement mode: {config.mode}")
    if config.min_confidence not in CONFIDENCE_ORDER:
        raise ValueError(f"unsupported min_confidence: {config.min_confidence}")
    if config.window_before_caption < 0 or config.window_after_caption < 0:
        raise ValueError("caption windows must be non-negative")
    if config.max_associated_blocks_per_caption < 1:
        raise ValueError("max_associated_blocks_per_caption must be >= 1")


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
