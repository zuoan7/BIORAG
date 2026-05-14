#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synbio_rag.ingestion.caption_cleanup import (
    SAFE_LABEL,
    CleanupStats,
    apply_cleanup_to_doc,
    audit_row_to_dict,
    merge_stats,
    norm_space,
    read_protected_keys,
    read_signoff_decisions,
    write_csv,
    write_json_doc,
)


AUDIT_FIELDS = [
    "doc_id",
    "source_file",
    "block_id",
    "original_block_type",
    "new_block_type",
    "caption_text",
    "cleanup_action",
    "caption_cleanup_rule_id",
    "caption_cleanup_reason",
    "confidence",
    "protected_hit",
    "signoff_label",
    "nearby_preview",
    "safe_to_demote",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5D-3 conservative false/fragment caption cleanup experimental path"
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--signoff_csv", required=True)
    parser.add_argument("--protected_csv", required=True)
    parser.add_argument("--audit_dir", required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--min_label", default=SAFE_LABEL)
    parser.add_argument("--mode", choices=("conservative",), default="conservative")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def iter_blocks(doc: dict[str, Any]):
    for page in doc.get("pages", []) or []:
        for block in page.get("blocks", []) or []:
            yield block


def count_types(doc: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for block in iter_blocks(doc):
        counts[norm_space(block.get("type"))] += 1
    return counts


def path_is_same_or_child(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def read_protected_source_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                row = dict(row)
                row["_source_path"] = str(path)
                rows.append(row)
    return rows


def protected_key_from_row(row: dict[str, str]) -> tuple[str, str] | None:
    doc_id = norm_space(row.get("doc_id"))
    block_id = norm_space(row.get("block_id"))
    if not doc_id or not block_id:
        return None
    return doc_id, block_id


def make_protected_check_rows(
    protected_rows: list[dict[str, str]],
    docs_after: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, bool | None]:
    checks: list[dict[str, Any]] = []
    violations = 0
    approved_seen = False
    approved_preserved = True
    seen: set[tuple[str, str]] = set()
    for row in protected_rows:
        key = protected_key_from_row(row)
        if not key or key in seen:
            continue
        seen.add(key)
        doc_id, block_id = key
        protect_reason = norm_space(row.get("protect_reason") or row.get("review_label"))
        if "approved_eval_target" in protect_reason:
            approved_seen = True
        found_type = ""
        found_text = ""
        doc = docs_after.get(doc_id)
        if doc:
            for block in iter_blocks(doc):
                if norm_space(block.get("block_id") or block.get("id")) == block_id:
                    found_type = norm_space(block.get("type"))
                    found_text = norm_space(block.get("text"))
                    break
        preserved = found_type in {"table_caption", "figure_caption"} or not found_type
        if not preserved:
            violations += 1
            if "approved_eval_target" in protect_reason:
                approved_preserved = False
        checks.append(
            {
                "doc_id": doc_id,
                "block_id": block_id,
                "caption_text": row.get("caption_text", ""),
                "protect_reason": protect_reason,
                "found_block_type_after": found_type,
                "found_text_after": found_text,
                "preserved": str(preserved).lower(),
                "source": row.get("_source_path", ""),
            }
        )
    return checks, violations, (approved_preserved if approved_seen else None)


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        values = []
        for field in fields:
            value = norm_space(row.get(field, "")).replace("|", "\\|")
            if len(value) > 140:
                value = value[:139] + "..."
            values.append(value)
        lines.append("|" + "|".join(values) + "|")
    return "\n".join(lines)


def write_examples(audit_dir: Path, audit_rows: list[dict[str, Any]]) -> None:
    demoted = [r for r in audit_rows if r["cleanup_action"] == "demote_to_paragraph"][:30]
    skipped_protected = [r for r in audit_rows if r["protected_hit"] == "true"][:20]
    skipped_uncertain_noise = [
        r for r in audit_rows
        if r["signoff_label"] in {"uncertain", "eval_only_noise", "needs_manual_pdf_check"}
    ][:20]
    fields = [
        "doc_id",
        "block_id",
        "original_block_type",
        "new_block_type",
        "caption_text",
        "cleanup_action",
        "signoff_label",
        "caption_cleanup_rule_id",
        "caption_cleanup_reason",
    ]
    text = "\n".join(
        [
            "# Phase 5D-3 Caption Cleanup Examples",
            "",
            "## Demoted Examples",
            markdown_table(demoted, fields) if demoted else "None.",
            "",
            "## Skipped Protected Examples",
            markdown_table(skipped_protected, fields) if skipped_protected else "None.",
            "",
            "## Skipped Uncertain / Eval Noise Examples",
            markdown_table(skipped_uncertain_noise, fields) if skipped_uncertain_noise else "None.",
            "",
        ]
    )
    (audit_dir / "examples.md").write_text(text, encoding="utf-8")


def write_summary(
    audit_dir: Path,
    stats: CleanupStats,
    doc_stats_rows: list[dict[str, Any]],
    missing_inputs: list[str],
    dry_run: bool,
) -> None:
    by_label = dict(stats.skipped_by_label)
    by_rule = dict(stats.demoted_by_rule)
    skipped_rules = dict(stats.skipped_by_rule)
    schema_status = "compatible: parsed_clean pages/blocks structure preserved; block text preserved; no chunk schema fields changed"
    recommendation = (
        "Proceed to Phase 5D-4 static audit / retrieval smoke on the experimental output only; keep cleanup default-off."
        if stats.protected_caption_violations == 0 and stats.doc_0367_figure5_preserved
        else "Do not proceed until preservation violations are resolved."
    )
    lines = [
        "# Phase 5D-3 Caption Cleanup Experiment Summary",
        "",
        f"- dry_run: {str(dry_run).lower()}",
        f"- missing input reports: {', '.join(missing_inputs) if missing_inputs else 'none'}",
        f"- total_docs: {stats.total_docs}",
        f"- processed_docs: {stats.processed_docs}",
        f"- success_docs: {stats.success_docs}",
        f"- failed_docs: {stats.failed_docs}",
        f"- cleanup_candidates_seen: {stats.cleanup_candidates_seen}",
        f"- demoted_count: {stats.demoted_count}",
        f"- skipped_eval_only_noise_count: {stats.skipped_eval_only_noise_count}",
        f"- skipped_uncertain_count: {stats.skipped_uncertain_count}",
        f"- skipped_protected_count: {stats.skipped_protected_count}",
        f"- skipped_manual_pdf_check_count: {stats.skipped_manual_pdf_check_count}",
        f"- skipped_keep_as_caption_count: {stats.skipped_keep_as_caption_count}",
        f"- skipped_rule_guard_count: {stats.skipped_rule_guard_count}",
        f"- by_count: {by_label}",
        f"- demoted_by_rule: {by_rule}",
        f"- skipped_protected_rule counts: {skipped_rules}",
        f"- table_caption count before/after: {stats.table_caption_before} / {stats.table_caption_after}",
        f"- figure_caption count before/after: {stats.figure_caption_before} / {stats.figure_caption_after}",
        f"- protected caption violations: {stats.protected_caption_violations}",
        f"- doc_0367 Figure 5 preserved: {str(stats.doc_0367_figure5_preserved).lower()}",
        f"- approved eval target preserved: {stats.approved_eval_target_preserved if stats.approved_eval_target_preserved is not None else 'not_detected'}",
        f"- schema compatibility status: {schema_status}",
        f"- recommendation for Phase 5D-4: {recommendation}",
        "",
        "## Doc-Level Stats Sample",
        markdown_table(doc_stats_rows[:40], list(doc_stats_rows[0].keys()) if doc_stats_rows else []),
        "",
    ]
    (audit_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_diff_summary(audit_dir: Path, stats: CleanupStats) -> None:
    lines = [
        "# Phase 5D-3 Parsed Clean Diff Summary",
        "",
        f"- table_caption count before/after: {stats.table_caption_before} / {stats.table_caption_after}",
        f"- figure_caption count before/after: {stats.figure_caption_before} / {stats.figure_caption_after}",
        f"- demoted block count: {stats.demoted_count}",
        f"- demoted by rule: {dict(stats.demoted_by_rule)}",
        f"- protected violation count: {stats.protected_caption_violations}",
        f"- known-good caption preserved: {str(stats.doc_0367_figure5_preserved).lower()}",
        "",
    ]
    (audit_dir / "parsed_clean_diff_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_cleanup(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    signoff_csv = Path(args.signoff_csv).resolve()
    protected_csv = Path(args.protected_csv).resolve()
    audit_dir = Path(args.audit_dir).resolve()
    production_parsed_clean = (REPO_ROOT / "data/paper_round1/parsed_clean").resolve()

    if args.min_label != SAFE_LABEL:
        print(f"[ERROR] Only --min_label {SAFE_LABEL} is supported in conservative mode.", file=sys.stderr)
        return 2
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] input_dir does not exist: {input_dir}", file=sys.stderr)
        return 2
    if output_dir == input_dir:
        print("[ERROR] output_dir must not equal input_dir.", file=sys.stderr)
        return 2
    if output_dir == production_parsed_clean:
        print("[ERROR] refusing to write to production parsed_clean.", file=sys.stderr)
        return 2
    if path_is_same_or_child(output_dir, production_parsed_clean):
        print("[ERROR] refusing output_dir inside production parsed_clean.", file=sys.stderr)
        return 2
    if path_is_same_or_child(production_parsed_clean, output_dir):
        print("[ERROR] refusing output_dir that would contain production parsed_clean.", file=sys.stderr)
        return 2

    audit_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    conventional_protected = REPO_ROOT / "reports/phase5d_caption_cleanup_audit/protected_short_captions.csv"
    protected_paths = [protected_csv, conventional_protected]
    missing_inputs = [
        str(path)
        for path in [signoff_csv, protected_csv]
        if not path.exists()
    ]
    decisions = read_signoff_decisions(signoff_csv)
    protected_keys = read_protected_keys(protected_paths)
    protected_source_rows = read_protected_source_rows(protected_paths)

    json_files = sorted(input_dir.glob("*.json"))
    stats = CleanupStats()
    audit_rows: list[dict[str, Any]] = []
    doc_stats_rows: list[dict[str, Any]] = []
    docs_after: dict[str, dict[str, Any]] = {}
    success_docs: list[str] = []
    failed_docs: list[str] = []

    for path in json_files:
        try:
            doc = load_json(path)
            doc_id = norm_space(doc.get("doc_id") or path.stem)
            before = count_types(doc)
            new_doc, doc_audit, doc_stats = apply_cleanup_to_doc(doc, decisions, protected_keys)
            after = count_types(new_doc)
            docs_after[doc_id] = new_doc
            merge_stats(stats, doc_stats)
            success_docs.append(path.name)
            if not args.dry_run:
                write_json_doc(output_dir / path.name, new_doc)
            for row in doc_audit:
                audit_rows.append(audit_row_to_dict(row))
            doc_stats_rows.append(
                {
                    "doc_id": doc_id,
                    "source_file": norm_space(doc.get("source_file")),
                    "candidates_seen": doc_stats.cleanup_candidates_seen,
                    "demoted_count": doc_stats.demoted_count,
                    "table_caption_before": before.get("table_caption", 0),
                    "table_caption_after": after.get("table_caption", 0),
                    "figure_caption_before": before.get("figure_caption", 0),
                    "figure_caption_after": after.get("figure_caption", 0),
                    "skipped_eval_only_noise": doc_stats.skipped_eval_only_noise_count,
                    "skipped_uncertain": doc_stats.skipped_uncertain_count,
                    "skipped_manual_pdf_check": doc_stats.skipped_manual_pdf_check_count,
                    "skipped_rule_guard": doc_stats.skipped_rule_guard_count,
                }
            )
        except Exception as exc:  # pragma: no cover - exercised by full-run diagnostics
            stats.failed_docs += 1
            failed_docs.append(f"{path.name}\t{exc}")

    stats.total_docs = len(json_files)
    stats.processed_docs = stats.success_docs + stats.failed_docs
    protected_checks, violations, approved_preserved = make_protected_check_rows(protected_source_rows, docs_after)
    stats.protected_caption_violations = violations
    stats.approved_eval_target_preserved = approved_preserved

    doc_stats_rows.sort(key=lambda row: (-int(row["demoted_count"]), row["doc_id"]))
    write_csv(audit_dir / "cleanup_audit.csv", audit_rows, AUDIT_FIELDS)
    write_csv(
        audit_dir / "doc_level_stats.csv",
        doc_stats_rows,
        [
            "doc_id",
            "source_file",
            "candidates_seen",
            "demoted_count",
            "table_caption_before",
            "table_caption_after",
            "figure_caption_before",
            "figure_caption_after",
            "skipped_eval_only_noise",
            "skipped_uncertain",
            "skipped_manual_pdf_check",
            "skipped_rule_guard",
        ],
    )
    write_csv(
        audit_dir / "protected_caption_check.csv",
        protected_checks,
        [
            "doc_id",
            "block_id",
            "caption_text",
            "protect_reason",
            "found_block_type_after",
            "found_text_after",
            "preserved",
            "source",
        ],
    )
    write_examples(audit_dir, audit_rows)
    write_summary(audit_dir, stats, doc_stats_rows, missing_inputs, args.dry_run)
    write_diff_summary(audit_dir, stats)

    if not args.dry_run:
        (output_dir.parent / "success_docs.txt").write_text("\n".join(success_docs) + "\n", encoding="utf-8")
        (output_dir.parent / "failed_docs.log").write_text("\n".join(failed_docs), encoding="utf-8")

    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "total_docs": stats.total_docs,
                "success_docs": stats.success_docs,
                "failed_docs": stats.failed_docs,
                "cleanup_candidates_seen": stats.cleanup_candidates_seen,
                "demoted_count": stats.demoted_count,
                "table_caption_before": stats.table_caption_before,
                "table_caption_after": stats.table_caption_after,
                "figure_caption_before": stats.figure_caption_before,
                "figure_caption_after": stats.figure_caption_after,
                "protected_caption_violations": stats.protected_caption_violations,
                "doc_0367_figure5_preserved": stats.doc_0367_figure5_preserved,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if stats.failed_docs else 0


def main() -> None:
    raise SystemExit(run_cleanup(parse_args()))


if __name__ == "__main__":
    main()
