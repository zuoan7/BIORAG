#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize the v5 full ingestion mini-regression."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SAMPLE_DOCS = [
    "doc_0005",
    "doc_0008",
    "doc_0037",
    "doc_0038",
    "doc_0083",
    "doc_0111",
    "doc_0163",
    "doc_0165",
    "doc_0200",
    "doc_0235",
    "doc_0250",
    "doc_0282",
    "doc_0287",
    "doc_0430",
    "doc_0566",
]

POLLUTION_PATTERNS = {
    "Journal Pre-proof": re.compile(r"Journal\s+Pre-proof", re.I),
    "This is a PDF file of an article": re.compile(r"This\s+is\s+a\s+PDF\s+file\s+of\s+an\s+article", re.I),
    "This is a PDF of an article": re.compile(r"This\s+is\s+a\s+PDF\s+of\s+an\s+article", re.I),
    "S1096-7176": re.compile(r"S1096-7176", re.I),
    "YMBEN": re.compile(r"\bYMBEN\b", re.I),
    "Correspondence:": re.compile(r"Correspondence\s*:", re.I),
    "University of Hawaii at Manoa Library": re.compile(r"University\s+of\s+Hawaii\s+at\s+Manoa\s+Library", re.I),
    "Page N of N": re.compile(r"Page\s+\d+\s+of\s+\d+", re.I),
    "OPEN ACCESS": re.compile(r"\bOPEN\s+ACCESS\b", re.I),
    "表达 Fam20C": re.compile(r"表达\s*Fam20C"),
    "[TABLE TEXT] was mixed": re.compile(r"\[TABLE TEXT\]\s+was\s+mixed", re.I),
    "[TABLE TEXT] incubated": re.compile(r"\[TABLE TEXT\]\s+incubated", re.I),
    "[TABLE TEXT] After identification": re.compile(r"\[TABLE TEXT\]\s+After\s+identification", re.I),
    "[TABLE TEXT] 48% byproduct": re.compile(r"\[TABLE TEXT\]\s+48%\s+byproduct", re.I),
    "[TABLE TEXT] gDW-1": re.compile(r"\[TABLE TEXT\]\s+gDW-1", re.I),
}

EVIDENCE_PATTERNS = {
    "[FIGURE CAPTION]": re.compile(r"\[FIGURE CAPTION\]"),
    "[TABLE CAPTION]": re.compile(r"\[TABLE CAPTION\]"),
    "[TABLE TEXT]": re.compile(r"\[TABLE TEXT\]"),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _preview(text: str, limit: int = 220) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:limit]


def _md(text: Any) -> str:
    return str(text if text is not None else "").replace("|", "\\|")


def _scan_chunks(path: Path) -> dict[str, Any]:
    pollution = {
        label: {"json_line_hit_count": 0, "text_retrieval_hit_count": 0, "examples": []}
        for label in POLLUTION_PATTERNS
    }
    evidence = {label: {"json_line_hit_count": 0, "examples": []} for label in EVIDENCE_PATTERNS}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            if not raw_line.strip():
                continue
            chunk = json.loads(raw_line)
            text = chunk.get("text", "") or ""
            retrieval_text = chunk.get("retrieval_text", "") or ""
            text_retrieval = f"{text}\n{retrieval_text}"
            for label, pattern in POLLUTION_PATTERNS.items():
                json_hit = bool(pattern.search(raw_line))
                text_hit = bool(pattern.search(text_retrieval))
                if json_hit:
                    pollution[label]["json_line_hit_count"] += 1
                if text_hit:
                    pollution[label]["text_retrieval_hit_count"] += 1
                    if len(pollution[label]["examples"]) < 5:
                        pollution[label]["examples"].append({
                            "line": line_no,
                            "doc_id": chunk.get("doc_id"),
                            "chunk_id": chunk.get("chunk_id"),
                            "page_start": chunk.get("page_start"),
                            "page_end": chunk.get("page_end"),
                            "text_preview": _preview(text),
                        })
            for label, pattern in EVIDENCE_PATTERNS.items():
                if pattern.search(raw_line):
                    evidence[label]["json_line_hit_count"] += 1
                    if len(evidence[label]["examples"]) < 5:
                        evidence[label]["examples"].append({
                            "line": line_no,
                            "doc_id": chunk.get("doc_id"),
                            "chunk_id": chunk.get("chunk_id"),
                            "text_preview": _preview(text),
                        })
    return {"pollution": pollution, "evidence": evidence}


def _summary(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("summary", {}) if isinstance(data, dict) else {}


def _command_rows() -> list[tuple[str, str]]:
    return [
        ("copy samples", "completed"),
        ("pdf_to_structured.py", "completed"),
        ("clean_parsed_structure.py", "completed"),
        ("preprocess_and_chunk.py", "completed"),
        ("validate_parsed_raw_v4.py", "completed"),
        ("audit_two_column_order_v5.py", "completed"),
        ("validate_parsed_clean_v4.py", "completed"),
        ("validate_chunks_v4.py", "completed"),
        ("generate_full_ingestion_spotcheck_v5.py", "completed"),
    ]


def _build_result(args: argparse.Namespace) -> dict[str, Any]:
    parser_json = _load_json(Path(args.parser_json))
    two_column_json = _load_json(Path(args.two_column_json))
    clean_json = _load_json(Path(args.clean_json))
    chunk_json = _load_json(Path(args.chunk_json))
    grep = _scan_chunks(Path(args.chunks_jsonl))

    parser_summary = _summary(parser_json)
    two_column_summary = _summary(two_column_json)
    clean_summary = _summary(clean_json)
    chunk_summary = _summary(chunk_json)

    risk_flags = two_column_summary.get("risk_flag_distribution", {}) or {}
    parser_fail_reasons: list[str] = []
    for flag in ("mid_anchor_displaced_after_columns", "single_column_interleaved_lr", "same_column_y_inversion"):
        if int(risk_flags.get(flag, 0) or 0) > 0:
            parser_fail_reasons.append(flag)
    if int(parser_summary.get("total_journal_preproof_noise_in_page_text_count", 0) or 0) > 0:
        parser_fail_reasons.append("journal_preproof_page_text")
    if int(parser_summary.get("total_journal_preproof_noise_in_block_text_count", 0) or 0) > 0:
        parser_fail_reasons.append("journal_preproof_block_text")
    if parser_summary.get("conclusion") == "fail":
        parser_fail_reasons.append("parser_validation_fail")

    chunk_text_pollution = {
        label: item
        for label, item in grep["pollution"].items()
        if int(item["text_retrieval_hit_count"]) > 0
    }

    fail_reasons: list[str] = []
    warning_reasons: list[str] = []
    missing_docs = [doc for doc in args.missing_docs.split(",") if doc]
    if missing_docs:
        fail_reasons.append("missing_input_pdfs")
    if parser_fail_reasons:
        fail_reasons.extend(f"parser:{reason}" for reason in parser_fail_reasons)
    if clean_summary.get("conclusion") == "fail":
        fail_reasons.append("clean_validation_fail")
    if chunk_summary.get("conclusion") == "fail":
        fail_reasons.append("chunk_validation_fail")
    if chunk_text_pollution:
        fail_reasons.append("targeted_grep_text_or_retrieval_pollution")
    if args.tests_status != "pass":
        fail_reasons.append("tests_not_passed")
    if args.py_compile_status != "pass":
        fail_reasons.append("py_compile_not_passed")

    for flag in ("likely_two_column_but_fallback", "figure_table_heavy_fallback", "caption_image_distance_risk", "mid_anchor_requires_region_check"):
        if int(risk_flags.get(flag, 0) or 0) > 0:
            warning_reasons.append(flag)
    line_only_hits = {
        label: item
        for label, item in grep["pollution"].items()
        if int(item["json_line_hit_count"]) > 0 and int(item["text_retrieval_hit_count"]) == 0
    }
    if line_only_hits:
        warning_reasons.append("targeted_grep_json_line_only_hits")

    conclusion = "fail" if fail_reasons else ("warning" if warning_reasons else "pass")
    return {
        "branch": args.branch,
        "commit": args.commit,
        "dirty_status": Path(args.dirty_status_file).read_text(encoding="utf-8") if args.dirty_status_file else "",
        "sample_docs": SAMPLE_DOCS,
        "missing_docs": missing_docs,
        "input_pdf_dir": args.input_pdf_dir,
        "parsed_raw_dir": args.parsed_raw_dir,
        "parsed_clean_dir": args.parsed_clean_dir,
        "chunks_jsonl": args.chunks_jsonl,
        "commands": _command_rows(),
        "parser_summary": parser_summary,
        "two_column_summary": two_column_summary,
        "clean_summary": clean_summary,
        "chunk_summary": chunk_summary,
        "targeted_grep": grep,
        "tests_status": args.tests_status,
        "py_compile_status": args.py_compile_status,
        "spotcheck_report": args.spotcheck_report,
        "conclusion": conclusion,
        "fail_reasons": fail_reasons,
        "warning_reasons": warning_reasons,
    }


def render_report(result: dict[str, Any]) -> str:
    lines: list[str] = ["# v5 Full Ingestion Mini Regression", ""]
    lines.append("## Repository")
    lines.append("")
    lines.append(f"- branch: `{result['branch']}`")
    lines.append(f"- commit: `{result['commit']}`")
    dirty = result["dirty_status"].strip()
    lines.append(f"- dirty working tree: `{dirty if dirty else 'clean'}`")
    lines.append("")
    lines.append("## Samples")
    lines.append("")
    lines.append(f"- sample_doc_count: {len(result['sample_docs'])}")
    lines.append(f"- docs: `{', '.join(result['sample_docs'])}`")
    lines.append(f"- missing_docs: `{', '.join(result['missing_docs']) if result['missing_docs'] else 'none'}`")
    lines.append("")
    lines.append("## Commands")
    lines.append("")
    for command, status in result["commands"]:
        lines.append(f"- {command}: {status}")
    lines.append("")
    lines.append("## Parser Summary")
    lines.append("")
    _append_summary(lines, result["parser_summary"], [
        "conclusion",
        "sample_count",
        "total_pages",
        "total_two_column_pages",
        "total_image_block_count",
        "total_figure_caption_candidate_count",
        "total_table_caption_candidate_count",
        "total_journal_preproof_noise_in_page_text_count",
        "total_journal_preproof_noise_in_block_text_count",
        "potential_regression_doc_count",
        "order_strategy_distribution",
    ])
    _append_summary(lines, result["two_column_summary"], [
        "risk_page_count",
        "order_strategy_distribution",
        "risk_flag_distribution",
    ])
    lines.append("")
    lines.append("## Clean Summary")
    lines.append("")
    _append_summary(lines, result["clean_summary"], [
        "conclusion",
        "sample_doc_count",
        "total_pages",
        "clean_type_text_count",
        "layout_metadata_retention_rate",
        "journal_preproof_noise_in_clean_page_text_count",
        "journal_preproof_noise_in_clean_block_text_count",
        "correspondence_as_paragraph_count",
        "correspondence_as_table_text_count",
        "marginal_banner_as_paragraph_count",
        "marginal_banner_as_table_text_count",
        "marginal_banner_as_references_count",
        "false_table_text_body_sentence_count",
        "front_matter_affiliation_as_paragraph_count",
        "clean_figure_caption_count",
        "clean_table_caption_count",
        "clean_table_text_count",
        "numbered_reference_as_paragraph_count",
        "potential_regression_doc_count",
    ])
    lines.append("")
    lines.append("## Chunk Summary")
    lines.append("")
    _append_summary(lines, result["chunk_summary"], [
        "conclusion",
        "doc_count",
        "chunk_count",
        "source_block_id_retention_rate",
        "layout_metadata_retention_rate",
        "chunks_with_figure_caption",
        "chunks_with_table_caption",
        "chunks_with_table_text",
        "caption_loss_count",
        "table_text_loss_count",
        "metadata_in_chunk_text_count",
        "noise_in_chunk_text_count",
        "image_in_chunk_text_count",
        "references_in_chunk_text_count",
        "journal_preproof_in_chunk_text_count",
        "correspondence_in_chunk_text_count",
        "marginal_banner_in_chunk_text_count",
        "cover_metadata_in_chunk_text_count",
        "running_header_footer_in_chunk_text_count",
        "annotation_noise_in_chunk_text_count",
        "false_table_text_body_sentence_count",
        "potential_regression_doc_count",
    ])
    lines.append("")
    lines.append("## Targeted Grep Summary")
    lines.append("")
    lines.append("| pattern | json_line_hits | text_or_retrieval_hits | first examples |")
    lines.append("|---|---:|---:|---|")
    for label, item in result["targeted_grep"]["pollution"].items():
        examples = "; ".join(
            f"{ex.get('doc_id')}:{ex.get('chunk_id')}"
            for ex in item["examples"][:3]
        ) or "-"
        lines.append(f"| {_md(label)} | {item['json_line_hit_count']} | {item['text_retrieval_hit_count']} | {_md(examples)} |")
    lines.append("")
    lines.append("### Evidence Markers")
    lines.append("")
    lines.append("| marker | json_line_hits | first examples |")
    lines.append("|---|---:|---|")
    for label, item in result["targeted_grep"]["evidence"].items():
        examples = "; ".join(
            f"{ex.get('doc_id')}:{ex.get('chunk_id')}"
            for ex in item["examples"][:3]
        ) or "-"
        lines.append(f"| {_md(label)} | {item['json_line_hit_count']} | {_md(examples)} |")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- conclusion: **{result['conclusion']}**")
    lines.append(f"- fail_reasons: `{', '.join(result['fail_reasons']) if result['fail_reasons'] else 'none'}`")
    lines.append(f"- warning_reasons: `{', '.join(result['warning_reasons']) if result['warning_reasons'] else 'none'}`")
    lines.append(f"- tests_status: `{result['tests_status']}`")
    lines.append(f"- py_compile_status: `{result['py_compile_status']}`")
    lines.append(f"- manual_spotcheck: `{result['spotcheck_report']}`")
    lines.append("")
    if result["conclusion"] == "pass":
        lines.append("Recommendation: can consider v5 Phase3b chunk retrieval smoke without rebuilding Milvus.")
    elif result["conclusion"] == "warning":
        lines.append("Recommendation: finish the listed manual spot checks before v5 Phase3b.")
    else:
        lines.append("Recommendation: do not enter Phase3b. Address the fail reasons first.")
    lines.append("")
    lines.append("## Explicitly Not Done")
    lines.append("")
    lines.append("- no Milvus rebuild")
    lines.append("- no retrieval/generation/reranker change")
    lines.append("- no BM25 change")
    lines.append("- no parent-child index")
    lines.append("- no OCR/pdfplumber")
    return "\n".join(lines) + "\n"


def _append_summary(lines: list[str], summary: dict[str, Any], keys: list[str]) -> None:
    for key in keys:
        if key in summary:
            lines.append(f"- {key}: `{summary[key]}`")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize v5 full ingestion mini regression")
    parser.add_argument("--parser_json", required=True)
    parser.add_argument("--two_column_json", required=True)
    parser.add_argument("--clean_json", required=True)
    parser.add_argument("--chunk_json", required=True)
    parser.add_argument("--chunks_jsonl", required=True)
    parser.add_argument("--input_pdf_dir", required=True)
    parser.add_argument("--parsed_raw_dir", required=True)
    parser.add_argument("--parsed_clean_dir", required=True)
    parser.add_argument("--spotcheck_report", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--dirty_status_file", required=True)
    parser.add_argument("--missing_docs", default="")
    parser.add_argument("--tests_status", choices=["pass", "fail"], required=True)
    parser.add_argument("--py_compile_status", choices=["pass", "fail"], required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    result = _build_result(args)

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result), encoding="utf-8")
    print(f"conclusion={result['conclusion']}")
    print(f"report={report_path}")
    print(f"json={json_path}")


if __name__ == "__main__":
    main()
