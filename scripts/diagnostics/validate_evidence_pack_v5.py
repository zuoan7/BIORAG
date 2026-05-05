#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate evidence_pack_v5 documents against parsed_clean inputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ingestion.document_cleaning_v5 import (
    CONTAMINATION_PATTERNS,
    EXCLUDED_BLOCK_TYPES,
    FORBIDDEN_EVIDENCE_TYPES,
    INCLUDED_BLOCK_TYPES,
    iter_clean_blocks,
    looks_like_false_table_text_body_sentence,
    normalize_text,
    source_block_id,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _preview(text: str, limit: int = 180) -> str:
    return normalize_text(text)[:limit]


def _block_key(block: dict[str, Any]) -> str:
    return str(source_block_id(block) or block.get("block_id") or _preview(block.get("text", ""), 120))


def _unit_key(unit: dict[str, Any]) -> str:
    metadata = unit.get("metadata", {}) or {}
    return str(metadata.get("source_clean_block_key") or unit.get("source_block_id") or unit.get("block_id") or "")


def _unit_example(unit: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "doc_id": unit.get("doc_id"),
        "evidence_id": unit.get("evidence_id"),
        "type": unit.get("type"),
        "page": unit.get("page"),
        "source_block_id": unit.get("source_block_id"),
        "block_id": unit.get("block_id"),
        "text_preview": _preview(unit.get("text", "")),
        "bbox": unit.get("bbox"),
        "column": unit.get("column"),
        "reading_order": unit.get("reading_order"),
        "matched_reason": reason,
    }


def analyze(clean_dir: Path, evidence_dir: Path) -> dict[str, Any]:
    clean_docs = [load_json(path) for path in sorted(clean_dir.glob("*.json"))]
    evidence_docs = [load_json(path) for path in sorted(evidence_dir.glob("*.json"))]
    evidence_by_doc = {doc.get("doc_id", Path(doc.get("source_file", "doc")).stem): doc for doc in evidence_docs}

    totals = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_doc: list[dict[str, Any]] = []
    potential_regression_docs = set()

    for clean_doc in clean_docs:
        doc_id = clean_doc.get("doc_id") or "unknown_doc"
        evidence_doc = evidence_by_doc.get(doc_id)
        if not evidence_doc:
            totals["missing_evidence_doc_count"] += 1
            potential_regression_docs.add(doc_id)
            examples["missing_evidence_doc_examples"].append({"doc_id": doc_id})
            continue

        clean_blocks = iter_clean_blocks(clean_doc)
        units = [
            unit for unit in evidence_doc.get("evidence_units", []) or []
            if isinstance(unit, dict)
        ]
        clean_type_counts = Counter(block.get("type", "unknown") for block in clean_blocks)
        unit_type_counts = Counter(unit.get("type", "unknown") for unit in units)
        image_audit_blocks = [
            block for block in evidence_doc.get("excluded_image_blocks", []) or []
            if isinstance(block, dict)
        ]

        included_blocks = [
            block for block in clean_blocks
            if block.get("type") in INCLUDED_BLOCK_TYPES and normalize_text(block.get("text", ""))
        ]
        expected_keys_by_type: dict[str, set[str]] = defaultdict(set)
        for block in included_blocks:
            expected_keys_by_type[block.get("type", "unknown")].add(_block_key(block))

        actual_keys_by_type: dict[str, set[str]] = defaultdict(set)
        actual_keys_by_clean_type: dict[str, set[str]] = defaultdict(set)
        table_text_demoted = 0
        for unit in units:
            actual_keys_by_type[unit.get("type", "unknown")].add(_unit_key(unit))
            metadata = unit.get("metadata", {}) or {}
            source_clean_type = metadata.get("source_clean_block_type") or unit.get("type", "unknown")
            actual_keys_by_clean_type[source_clean_type].add(_unit_key(unit))
            if source_clean_type == "table_text" and unit.get("type") == "paragraph":
                table_text_demoted += 1

        forbidden = [unit for unit in units if unit.get("type") in FORBIDDEN_EVIDENCE_TYPES]
        missing_source = [unit for unit in units if not unit.get("source_block_id")]
        missing_layout = [
            unit for unit in units
            if unit.get("type") not in {"title", "section_heading", "subsection_heading"}
            and not (isinstance(unit.get("bbox"), list) and unit.get("column") is not None)
        ]
        contamination = []
        false_table_text = []
        for unit in units:
            for name, pattern in CONTAMINATION_PATTERNS.items():
                if pattern.search(unit.get("text", "")):
                    contamination.append(_unit_example(unit, name))
                    break
            if unit.get("type") == "table_text" and looks_like_false_table_text_body_sentence(unit.get("text", "")):
                false_table_text.append(_unit_example(unit, "false_table_text_body_sentence"))

        losses: dict[str, list[str]] = {}
        for block_type, expected_keys in expected_keys_by_type.items():
            lost = sorted(key for key in expected_keys if key not in actual_keys_by_clean_type.get(block_type, set()))
            if lost:
                losses[block_type] = lost

        for unit in forbidden[:20]:
            examples["forbidden_evidence_examples"].append(_unit_example(unit, "forbidden_evidence_type"))
        for unit in missing_source[:20]:
            examples["missing_source_block_id_examples"].append(_unit_example(unit, "missing_source_block_id"))
        for unit in missing_layout[:20]:
            examples["missing_layout_metadata_examples"].append(_unit_example(unit, "missing_layout_metadata"))
        examples["contamination_examples"].extend(contamination[:20])
        examples["false_table_text_body_sentence_examples"].extend(false_table_text[:20])
        for block_type in ("figure_caption", "table_caption", "table_text"):
            for key in losses.get(block_type, [])[:20]:
                examples[f"{block_type}_loss_examples"].append({"doc_id": doc_id, "source_block_key": key})

        table_caption_groups = {
            unit.get("table_group_id")
            for unit in units
            if unit.get("type") == "table_caption" and unit.get("table_group_id")
        }
        table_text_groups = {
            unit.get("table_group_id")
            for unit in units
            if unit.get("type") == "table_text" and unit.get("table_group_id")
        }

        doc_summary = {
            "doc_id": doc_id,
            "clean_block_count": len(clean_blocks),
            "included_clean_block_count": len(included_blocks),
            "evidence_unit_count": len(units),
            "clean_type_counts": dict(clean_type_counts),
            "evidence_type_counts": dict(unit_type_counts),
            "forbidden_evidence_unit_count": len(forbidden),
            "missing_source_block_id_count": len(missing_source),
            "missing_layout_metadata_count": len(missing_layout),
            "contamination_count": len(contamination),
            "false_table_text_body_sentence_count": len(false_table_text),
            "loss_counts": {k: len(v) for k, v in losses.items()},
            "table_caption_group_count": len(table_caption_groups),
            "table_text_group_count": len(table_text_groups),
            "table_text_demoted_to_paragraph_count": table_text_demoted,
            "image_clean_count": clean_type_counts.get("image", 0),
            "image_audit_block_count": len(image_audit_blocks),
        }
        per_doc.append(doc_summary)

        totals["doc_count"] += 1
        totals["clean_block_count"] += len(clean_blocks)
        totals["included_clean_block_count"] += len(included_blocks)
        totals["evidence_unit_count"] += len(units)
        totals["forbidden_evidence_unit_count"] += len(forbidden)
        totals["missing_source_block_id_count"] += len(missing_source)
        totals["missing_layout_metadata_count"] += len(missing_layout)
        totals["contamination_count"] += len(contamination)
        totals["false_table_text_body_sentence_count"] += len(false_table_text)
        totals["figure_caption_clean_count"] += clean_type_counts.get("figure_caption", 0)
        totals["table_caption_clean_count"] += clean_type_counts.get("table_caption", 0)
        totals["table_text_clean_count"] += clean_type_counts.get("table_text", 0)
        totals["figure_caption_evidence_count"] += unit_type_counts.get("figure_caption", 0)
        totals["table_caption_evidence_count"] += unit_type_counts.get("table_caption", 0)
        totals["table_text_evidence_count"] += unit_type_counts.get("table_text", 0)
        totals["table_text_demoted_to_paragraph_count"] += table_text_demoted
        totals["image_audit_block_count"] += len(image_audit_blocks)
        for block_type in EXCLUDED_BLOCK_TYPES:
            totals[f"{block_type}_clean_count"] += clean_type_counts.get(block_type, 0)
        for block_type, lost in losses.items():
            totals[f"{block_type}_loss_count"] += len(lost)

        if forbidden or contamination or losses or len(missing_source) / max(len(units), 1) > 0.05:
            potential_regression_docs.add(doc_id)
        if false_table_text:
            potential_regression_docs.add(doc_id)
        if clean_type_counts.get("image", 0) != len(image_audit_blocks):
            potential_regression_docs.add(doc_id)
            examples["image_audit_mismatch_examples"].append({
                "doc_id": doc_id,
                "image_clean_count": clean_type_counts.get("image", 0),
                "image_audit_block_count": len(image_audit_blocks),
            })

    evidence_unit_count = totals["evidence_unit_count"]
    summary = dict(totals)
    summary["source_block_id_retention_rate"] = (
        1.0 - totals["missing_source_block_id_count"] / evidence_unit_count
        if evidence_unit_count else 1.0
    )
    summary["layout_metadata_retention_rate"] = (
        1.0 - totals["missing_layout_metadata_count"] / evidence_unit_count
        if evidence_unit_count else 1.0
    )
    summary["potential_regression_doc_count"] = len(potential_regression_docs)
    summary["potential_regression_docs"] = sorted(potential_regression_docs)
    summary["examples"] = {key: value[:50] for key, value in examples.items()}
    summary["per_doc"] = per_doc
    summary["status"] = "pass" if _is_pass(summary) else "fail"
    return summary


def _is_pass(summary: dict[str, Any]) -> bool:
    return (
        summary.get("missing_evidence_doc_count", 0) == 0
        and summary.get("forbidden_evidence_unit_count", 0) == 0
        and summary.get("contamination_count", 0) == 0
        and summary.get("false_table_text_body_sentence_count", 0) == 0
        and summary.get("figure_caption_loss_count", 0) == 0
        and summary.get("table_caption_loss_count", 0) == 0
        and summary.get("image_audit_block_count", 0) == summary.get("image_clean_count", 0)
        and summary.get("source_block_id_retention_rate", 0.0) >= 0.95
        and summary.get("layout_metadata_retention_rate", 0.0) >= 0.95
    )


def write_report(summary: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# v5 Evidence Pack Validation",
        "",
        f"Status: **{summary.get('status')}**",
        "",
        "## Summary",
        "",
        f"- doc_count: {summary.get('doc_count', 0)}",
        f"- clean_block_count: {summary.get('clean_block_count', 0)}",
        f"- included_clean_block_count: {summary.get('included_clean_block_count', 0)}",
        f"- evidence_unit_count: {summary.get('evidence_unit_count', 0)}",
        f"- forbidden_evidence_unit_count: {summary.get('forbidden_evidence_unit_count', 0)}",
        f"- contamination_count: {summary.get('contamination_count', 0)}",
        f"- false_table_text_body_sentence_count: {summary.get('false_table_text_body_sentence_count', 0)}",
        f"- source_block_id_retention_rate: {summary.get('source_block_id_retention_rate', 0.0):.4f}",
        f"- layout_metadata_retention_rate: {summary.get('layout_metadata_retention_rate', 0.0):.4f}",
        f"- figure_caption: {summary.get('figure_caption_clean_count', 0)} -> {summary.get('figure_caption_evidence_count', 0)}",
        f"- table_caption: {summary.get('table_caption_clean_count', 0)} -> {summary.get('table_caption_evidence_count', 0)}",
        f"- table_text: {summary.get('table_text_clean_count', 0)} -> {summary.get('table_text_evidence_count', 0)}",
        f"- table_text_demoted_to_paragraph_count: {summary.get('table_text_demoted_to_paragraph_count', 0)}",
        f"- metadata/noise/image/references excluded: "
        f"{summary.get('metadata_clean_count', 0)}/"
        f"{summary.get('noise_clean_count', 0)}/"
        f"{summary.get('image_clean_count', 0)}/"
        f"{summary.get('references_clean_count', 0)}",
        f"- image_audit_block_count: {summary.get('image_audit_block_count', 0)}",
        f"- potential_regression_doc_count: {summary.get('potential_regression_doc_count', 0)}",
        "",
        "## Examples",
        "",
    ]
    examples = summary.get("examples", {}) or {}
    if not examples:
        lines.append("No blocking examples.")
    else:
        for name, items in examples.items():
            if not items:
                continue
            lines.append(f"### {name}")
            for item in items[:10]:
                lines.append(f"- `{json.dumps(item, ensure_ascii=False)}`")
            lines.append("")
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate evidence_pack_v5 output")
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--evidence_dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    summary = analyze(Path(args.clean_dir), Path(args.evidence_dir))
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, Path(args.report))
    print(f"status={summary['status']}")
    print(f"doc_count={summary.get('doc_count', 0)}")
    print(f"evidence_unit_count={summary.get('evidence_unit_count', 0)}")
    print(f"forbidden_evidence_unit_count={summary.get('forbidden_evidence_unit_count', 0)}")
    print(f"contamination_count={summary.get('contamination_count', 0)}")


if __name__ == "__main__":
    main()
