#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample context-aware cleaning rule hits for manual false-positive review."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


DEFAULT_CONTEXT_RULE_IDS = [
    "context_affiliation_address",
    "context_author_contribution",
    "context_back_matter_metadata",
    "context_correspondence_metadata",
    "context_body_fragment_heading",
    "context_running_header_by_position",
]

PREVIEW_LIMIT = 240
SAMPLE_SEED = 13


def _preview(text: Any, limit: int = PREVIEW_LIMIT) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    return compact[:limit]


def _parse_rule_ids(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_CONTEXT_RULE_IDS)
    rule_ids = [item.strip() for item in value.split(",") if item.strip()]
    return rule_ids or list(DEFAULT_CONTEXT_RULE_IDS)


def _iter_json_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("*.json") if path.is_file())


def _iter_blocks_with_neighbors(data: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]]:
    blocks = [
        block
        for page in data.get("pages", []) or []
        if isinstance(page, dict)
        for block in page.get("blocks", []) or []
        if isinstance(block, dict)
    ]
    result = []
    for idx, block in enumerate(blocks):
        prev_block = blocks[idx - 1] if idx > 0 else None
        next_block = blocks[idx + 1] if idx + 1 < len(blocks) else None
        result.append((block, prev_block, next_block))
    return result


def _sample_record(
    data: dict[str, Any],
    path: Path,
    block: dict[str, Any],
    prev_block: dict[str, Any] | None,
    next_block: dict[str, Any] | None,
    rule_id: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "doc_id": data.get("doc_id", path.stem),
        "source_file": data.get("source_file", ""),
        "page": block.get("page"),
        "block_id": block.get("block_id"),
        "block_type": block.get("type"),
        "cleaning_rule_id": rule_id,
        "cleaning_reason": reason,
        "text_preview": _preview(block.get("text", "")),
        "previous_block_preview": _preview(prev_block.get("text", "")) if prev_block else "",
        "next_block_preview": _preview(next_block.get("text", "")) if next_block else "",
        "section_path": block.get("section_path", []) or [],
    }


def collect_context_rule_hits(input_dir: Path, rule_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Collect all parsed_clean block hits for the requested rule IDs."""
    wanted = set(rule_ids)
    hits: dict[str, list[dict[str, Any]]] = {rule_id: [] for rule_id in rule_ids}
    for path in _iter_json_files(input_dir):
        data = json.loads(path.read_text(encoding="utf-8"))
        for block, prev_block, next_block in _iter_blocks_with_neighbors(data):
            metadata = block.get("metadata", {}) or {}
            if not isinstance(metadata, dict):
                continue
            rule_id = metadata.get("cleaning_rule_id")
            if rule_id not in wanted:
                continue
            reason = str(metadata.get("cleaning_reason", "") or "")
            hits[str(rule_id)].append(_sample_record(data, path, block, prev_block, next_block, str(rule_id), reason))
    return hits


def sample_hits(hits: dict[str, list[dict[str, Any]]], sample_per_rule: int) -> dict[str, list[dict[str, Any]]]:
    """Return stable random samples per rule ID."""
    rng = random.Random(SAMPLE_SEED)
    samples: dict[str, list[dict[str, Any]]] = {}
    for rule_id, records in hits.items():
        ordered = sorted(
            records,
            key=lambda item: (
                str(item.get("doc_id", "")),
                int(item.get("page") or 0),
                str(item.get("block_id", "")),
                item.get("text_preview", ""),
            ),
        )
        if len(ordered) <= sample_per_rule:
            samples[rule_id] = ordered
        else:
            samples[rule_id] = sorted(
                rng.sample(ordered, sample_per_rule),
                key=lambda item: (
                    str(item.get("doc_id", "")),
                    int(item.get("page") or 0),
                    str(item.get("block_id", "")),
                    item.get("text_preview", ""),
                ),
            )
    return samples


def build_report(input_dir: Path, rule_ids: list[str], sample_per_rule: int) -> dict[str, Any]:
    hits = collect_context_rule_hits(input_dir, rule_ids)
    samples = sample_hits(hits, sample_per_rule)
    total_hits = {rule_id: len(records) for rule_id, records in hits.items()}
    sample_counts = {rule_id: len(records) for rule_id, records in samples.items()}
    return {
        "input_dir": str(input_dir),
        "rule_ids": rule_ids,
        "sample_per_rule": sample_per_rule,
        "total_hits_by_rule": total_hits,
        "sample_counts_by_rule": sample_counts,
        "samples_by_rule": samples,
    }


def _render_sample(sample: dict[str, Any], index: int) -> list[str]:
    section_path = sample.get("section_path") or []
    section_text = " > ".join(str(item) for item in section_path)
    lines = [
        f"### {index}. {sample.get('doc_id')} page {sample.get('page')} block {sample.get('block_id')}",
        "",
        f"- source_file: `{sample.get('source_file', '')}`",
        f"- block_type: `{sample.get('block_type', '')}`",
        f"- cleaning_reason: `{sample.get('cleaning_reason', '')}`",
        f"- section_path: `{section_text}`",
        "",
        "**Text**",
        "",
        f"> {sample.get('text_preview', '')}",
        "",
    ]
    if sample.get("previous_block_preview"):
        lines.extend(["**Previous Block**", "", f"> {sample['previous_block_preview']}", ""])
    if sample.get("next_block_preview"):
        lines.extend(["**Next Block**", "", f"> {sample['next_block_preview']}", ""])
    lines.append("---")
    lines.append("")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Context Rule False Positive Samples",
        "",
        f"- Input directory: `{report['input_dir']}`",
        f"- Sample per rule: {report['sample_per_rule']}",
        "",
        "## Counts",
        "",
    ]
    for rule_id in report["rule_ids"]:
        total = report["total_hits_by_rule"].get(rule_id, 0)
        sampled = report["sample_counts_by_rule"].get(rule_id, 0)
        lines.append(f"- `{rule_id}`: sampled {sampled} of {total}")
    lines.append("")

    samples_by_rule = report["samples_by_rule"]
    for rule_id in report["rule_ids"]:
        lines.extend([f"## `{rule_id}`", ""])
        samples = samples_by_rule.get(rule_id, [])
        if not samples:
            lines.extend(["No samples.", ""])
            continue
        for idx, sample in enumerate(samples, start=1):
            lines.extend(_render_sample(sample, idx))
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "context_rule_samples.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "context_rule_samples.md").write_text(render_markdown(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample context-aware cleaning rule hits for manual review.")
    parser.add_argument("--input_dir", required=True, type=Path, help="Directory containing parsed_clean JSON files.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory for context_rule_samples reports.")
    parser.add_argument("--rule_ids", help="Comma-separated cleaning_rule_id values to sample.")
    parser.add_argument("--sample_per_rule", type=int, default=20, help="Maximum samples per rule ID.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_per_rule < 1:
        raise SystemExit("--sample_per_rule must be >= 1")
    rule_ids = _parse_rule_ids(args.rule_ids)
    report = build_report(args.input_dir, rule_ids, args.sample_per_rule)
    write_report(report, args.output_dir)
    print(f"Scanned {args.input_dir}")
    print(f"Wrote {args.output_dir / 'context_rule_samples.json'}")
    print(f"Wrote {args.output_dir / 'context_rule_samples.md'}")
    for rule_id in rule_ids:
        sampled = report["sample_counts_by_rule"].get(rule_id, 0)
        total = report["total_hits_by_rule"].get(rule_id, 0)
        print(f"{rule_id}: sampled {sampled} of {total}")


if __name__ == "__main__":
    main()
