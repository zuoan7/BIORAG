#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit cleaning rule hits in parsed_clean and evidence-pack JSON files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def iter_json_files(input_dir: Path) -> list[Path]:
    """Return JSON files under input_dir in stable order."""
    return sorted(path for path in input_dir.glob("*.json") if path.is_file())


def iter_blocks(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return parsed_clean-style page blocks from a document."""
    return [
        block
        for page in data.get("pages", []) or []
        if isinstance(page, dict)
        for block in page.get("blocks", []) or []
        if isinstance(block, dict)
    ]


def iter_exclusion_examples(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return evidence exclusion audit examples from a document."""
    examples = data.get("excluded_block_examples", []) or []
    return [item for item in examples if isinstance(item, dict)]


def _counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"compare_to file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def audit_file(path: Path) -> dict[str, Any]:
    """Collect rule and block-type hit counts from one JSON document."""
    data = json.loads(path.read_text(encoding="utf-8"))
    blocks = iter_blocks(data)
    examples = iter_exclusion_examples(data)

    block_type_counts: Counter[str] = Counter()
    cleaning_rule_id_counts: Counter[str] = Counter()
    cleaning_reason_counts: Counter[str] = Counter()
    evidence_rule_id_counts: Counter[str] = Counter()
    evidence_reason_counts: Counter[str] = Counter()

    for block in blocks:
        block_type_counts[str(block.get("type", "unknown"))] += 1
        metadata = block.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            continue
        rule_id = metadata.get("cleaning_rule_id")
        reason = metadata.get("cleaning_reason")
        if rule_id:
            cleaning_rule_id_counts[str(rule_id)] += 1
        if reason:
            cleaning_reason_counts[str(reason)] += 1

    for item in examples:
        rule_id = item.get("rule_id")
        reason = item.get("reason") or item.get("matched_reason")
        if rule_id:
            evidence_rule_id_counts[str(rule_id)] += 1
        if reason:
            evidence_reason_counts[str(reason)] += 1

    return {
        "path": str(path),
        "doc_id": data.get("doc_id", path.stem),
        "parser_stage": data.get("parser_stage", ""),
        "block_count": len(blocks),
        "evidence_exclusion_example_count": len(examples),
        "block_type_counts": _counter_to_dict(block_type_counts),
        "cleaning_rule_id_counts": _counter_to_dict(cleaning_rule_id_counts),
        "cleaning_reason_counts": _counter_to_dict(cleaning_reason_counts),
        "evidence_exclusion_rule_id_counts": _counter_to_dict(evidence_rule_id_counts),
        "evidence_exclusion_reason_counts": _counter_to_dict(evidence_reason_counts),
    }


def aggregate_file_results(file_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-document audit results into a report payload."""
    totals = {
        "block_type_counts": Counter(),
        "cleaning_rule_id_counts": Counter(),
        "cleaning_reason_counts": Counter(),
        "evidence_exclusion_rule_id_counts": Counter(),
        "evidence_exclusion_reason_counts": Counter(),
    }
    docs_with_cleaning_rules: set[str] = set()
    docs_with_evidence_rules: set[str] = set()

    for result in file_results:
        doc_id = str(result.get("doc_id", ""))
        for key, counter in totals.items():
            counter.update(result.get(key, {}) or {})
        if result.get("cleaning_rule_id_counts"):
            docs_with_cleaning_rules.add(doc_id)
        if result.get("evidence_exclusion_rule_id_counts"):
            docs_with_evidence_rules.add(doc_id)

    return {
        "document_count": len(file_results),
        "total_block_count": sum(int(item.get("block_count", 0) or 0) for item in file_results),
        "total_evidence_exclusion_example_count": sum(
            int(item.get("evidence_exclusion_example_count", 0) or 0)
            for item in file_results
        ),
        "documents_with_cleaning_rule_id": len(docs_with_cleaning_rules),
        "documents_with_evidence_rule_id": len(docs_with_evidence_rules),
        "block_type_counts": _counter_to_dict(totals["block_type_counts"]),
        "cleaning_rule_id_counts": _counter_to_dict(totals["cleaning_rule_id_counts"]),
        "cleaning_reason_counts": _counter_to_dict(totals["cleaning_reason_counts"]),
        "evidence_exclusion_rule_id_counts": _counter_to_dict(totals["evidence_exclusion_rule_id_counts"]),
        "evidence_exclusion_reason_counts": _counter_to_dict(totals["evidence_exclusion_reason_counts"]),
        "files": file_results,
    }


def _render_counter_section(title: str, counts: dict[str, int]) -> list[str]:
    lines = [f"## {title}", ""]
    if not counts:
        lines.extend(["No hits.", ""])
        return lines
    lines.extend(f"- `{key}`: {value}" for key, value in counts.items())
    lines.append("")
    return lines


def _diff_counts(before: dict[str, int], after: dict[str, int]) -> dict[str, Any]:
    before_keys = set(before)
    after_keys = set(after)
    all_keys = sorted(before_keys | after_keys)
    changes = {
        key: {
            "before": int(before.get(key, 0) or 0),
            "after": int(after.get(key, 0) or 0),
            "delta": int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0),
        }
        for key in all_keys
        if int(before.get(key, 0) or 0) != int(after.get(key, 0) or 0)
    }
    return {
        "added": sorted(after_keys - before_keys),
        "removed": sorted(before_keys - after_keys),
        "changes": changes,
    }


def build_diff_report(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Build a stable diff between two audit reports."""
    diff_keys = [
        "cleaning_rule_id_counts",
        "block_type_counts",
        "cleaning_reason_counts",
        "evidence_exclusion_rule_id_counts",
        "evidence_exclusion_reason_counts",
    ]
    metrics = {
        key: _diff_counts(before.get(key, {}) or {}, after.get(key, {}) or {})
        for key in diff_keys
    }
    return {
        "before_document_count": before.get("document_count", 0),
        "after_document_count": after.get("document_count", 0),
        "before_total_block_count": before.get("total_block_count", 0),
        "after_total_block_count": after.get("total_block_count", 0),
        "added_rule_ids": metrics["cleaning_rule_id_counts"]["added"],
        "removed_rule_ids": metrics["cleaning_rule_id_counts"]["removed"],
        "metrics": metrics,
    }


def _render_diff_metric(title: str, metric: dict[str, Any]) -> list[str]:
    lines = [f"## {title}", ""]
    if metric["added"]:
        lines.append("Added: " + ", ".join(f"`{item}`" for item in metric["added"]))
    if metric["removed"]:
        lines.append("Removed: " + ", ".join(f"`{item}`" for item in metric["removed"]))
    if metric["changes"]:
        lines.extend(
            f"- `{key}`: {values['before']} -> {values['after']} ({values['delta']:+d})"
            for key, values in metric["changes"].items()
        )
    if not metric["added"] and not metric["removed"] and not metric["changes"]:
        lines.append("No changes.")
    lines.append("")
    return lines


def render_diff_summary(diff: dict[str, Any], compare_to: Path, input_dir: Path) -> str:
    """Render Markdown for an audit diff report."""
    lines = [
        "# Cleaning Rule Audit Diff",
        "",
        f"- Baseline report: `{compare_to}`",
        f"- After input directory: `{input_dir}`",
        f"- Documents: {diff['before_document_count']} -> {diff['after_document_count']}",
        f"- Blocks: {diff['before_total_block_count']} -> {diff['after_total_block_count']}",
        f"- Added cleaning rule IDs: {len(diff['added_rule_ids'])}",
        f"- Removed cleaning rule IDs: {len(diff['removed_rule_ids'])}",
        "",
    ]
    metric_titles = {
        "cleaning_rule_id_counts": "Cleaning Rule ID Changes",
        "block_type_counts": "Block Type Changes",
        "cleaning_reason_counts": "Cleaning Reason Changes",
        "evidence_exclusion_rule_id_counts": "Evidence Exclusion Rule ID Changes",
        "evidence_exclusion_reason_counts": "Evidence Exclusion Reason Changes",
    }
    for key, title in metric_titles.items():
        lines.extend(_render_diff_metric(title, diff["metrics"][key]))
    return "\n".join(lines).rstrip() + "\n"


def render_summary(report: dict[str, Any], input_dir: Path) -> str:
    """Render a compact Markdown audit summary."""
    lines = [
        "# Cleaning Rule Audit",
        "",
        f"- Input directory: `{input_dir}`",
        f"- Documents scanned: {report['document_count']}",
        f"- Blocks scanned: {report['total_block_count']}",
        f"- Evidence exclusion examples scanned: {report['total_evidence_exclusion_example_count']}",
        f"- Documents with block cleaning_rule_id: {report['documents_with_cleaning_rule_id']}",
        f"- Documents with evidence exclusion rule_id: {report['documents_with_evidence_rule_id']}",
        "",
    ]
    lines.extend(_render_counter_section("Block Types", report["block_type_counts"]))
    lines.extend(_render_counter_section("Block Cleaning Rule IDs", report["cleaning_rule_id_counts"]))
    lines.extend(_render_counter_section("Block Cleaning Reasons", report["cleaning_reason_counts"]))
    lines.extend(_render_counter_section("Evidence Exclusion Rule IDs", report["evidence_exclusion_rule_id_counts"]))
    lines.extend(_render_counter_section("Evidence Exclusion Reasons", report["evidence_exclusion_reason_counts"]))
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], output_dir: Path, input_dir: Path) -> None:
    """Write JSON and Markdown audit reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rule_hits.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(render_summary(report, input_dir), encoding="utf-8")


def write_diff_report(diff: dict[str, Any], output_dir: Path, compare_to: Path, input_dir: Path) -> None:
    """Write audit diff JSON and Markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rule_diff.json").write_text(
        json.dumps(diff, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(render_diff_summary(diff, compare_to, input_dir), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit BIORAG cleaning rule hits.")
    parser.add_argument("--input_dir", required=True, type=Path, help="Directory containing parsed_clean/evidence JSON files.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory where rule_hits.json and summary.md are written.")
    parser.add_argument("--compare_to", type=Path, help="Optional baseline rule_hits.json to diff against.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    before = None
    if args.compare_to:
        try:
            before = _load_report(args.compare_to)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
    files = iter_json_files(input_dir)
    file_results = [audit_file(path) for path in files]
    report = aggregate_file_results(file_results)
    write_report(report, output_dir, input_dir)
    print(f"Scanned {len(files)} JSON files")
    print(f"Wrote {output_dir / 'rule_hits.json'}")
    print(f"Wrote {output_dir / 'summary.md'}")
    if args.compare_to and before is not None:
        diff = build_diff_report(before, report)
        write_diff_report(diff, output_dir, args.compare_to, input_dir)
        print(f"Wrote {output_dir / 'rule_diff.json'}")
        print(f"Wrote diff summary {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
