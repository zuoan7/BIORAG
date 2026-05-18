#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.evaluate_ragas import (
    build_failure_diagnostics,
    build_raw_records,
    evaluate_retrieval,
    get_effective_final_answer_mode,
    get_generation_v2_debug,
    get_refusal_reason,
    load_records,
)
from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.schemas import QueryFilters

def normalize_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_settings(group_key: str) -> Settings:
    settings = Settings.from_env()
    settings.audit.enabled = False
    settings.generation.version = "old"
    settings.generation.v2_use_external_tools = False
    settings.generation.v2_use_history = False
    if group_key == "old_neighbor_on":
        settings.retrieval.neighbor_expansion_enabled = True
    elif group_key == "old_neighbor_off":
        settings.retrieval.neighbor_expansion_enabled = False
    elif group_key == "v2_seed_only":
        settings.generation.version = "v2"
        settings.retrieval.neighbor_expansion_enabled = True
    else:
        raise ValueError(group_key)
    return settings


def build_contexts(citations: list[dict[str, Any]]) -> list[str]:
    contexts: list[str] = []
    for citation in citations:
        quote = (citation.get("quote") or "").strip()
        title = (citation.get("title") or "").strip()
        section = (citation.get("section") or "").strip()
        source_file = (citation.get("source_file") or "").strip()
        context_text = quote or title
        if section:
            context_text = f"{section}: {context_text}" if context_text else section
        if source_file:
            context_text = f"{source_file} | {context_text}" if context_text else source_file
        if context_text:
            contexts.append(context_text)
    return contexts


def mode_bucket(mode: str) -> str:
    lowered = (mode or "").strip().lower()
    if lowered == "refuse":
        return "refuse"
    if "partial" in lowered:
        return "partial"
    return "full"


def citation_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in records:
        counter[str(normalize_int((item.get("retrieval_eval") or {}).get("citation_count")))] += 1
    return dict(sorted(counter.items(), key=lambda entry: int(entry[0])))


def support_pack_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in records:
        counter[str(normalize_int((item.get("raw_record") or {}).get("support_pack_count")))] += 1
    return dict(sorted(counter.items(), key=lambda entry: int(entry[0])))


def answer_mode_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in records:
        counter[str((item.get("raw_record") or {}).get("answer_mode") or "unknown")] += 1
    return dict(sorted(counter.items()))


def validator_flag_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in records:
        validator_debug = ((item.get("raw_record") or {}).get("validator_debug") or {})
        for key, value in validator_debug.items():
            if isinstance(value, bool) and value:
                counter[key] += 1
    return dict(sorted(counter.items()))


def failure_category(item: dict[str, Any]) -> str:
    retrieval_eval = item.get("retrieval_eval") or {}
    if retrieval_eval.get("strict_route_match") is False:
        return "route_mismatch"
    if retrieval_eval.get("refusal_no_citation"):
        return "refusal_no_citation"
    if retrieval_eval.get("zero_citation_substantive_answer"):
        return "zero_citation_substantive_answer"
    doc_metrics = retrieval_eval.get("doc_id_metrics") or {}
    if doc_metrics.get("expected") and not doc_metrics.get("hit"):
        return "doc_miss"
    section_metrics = retrieval_eval.get("section_metrics") or {}
    if section_metrics.get("expected") and not section_metrics.get("hit"):
        return "section_miss"
    evidence_metrics = retrieval_eval.get("evidence_metrics") or {}
    if isinstance(evidence_metrics.get("evidence_coverage"), (int, float)) and float(evidence_metrics.get("evidence_coverage")) == 0.0:
        return "evidence_not_supported_by_citations"
    mode = str((item.get("raw_record") or {}).get("answer_mode") or "")
    if mode_bucket(mode) == "partial":
        return "partial_answer"
    if mode_bucket(mode) == "refuse":
        return "refusal_other"
    return "ok"


def zero_citation_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for item in records:
        retrieval_eval = item.get("retrieval_eval") or {}
        if normalize_int(retrieval_eval.get("citation_count")) != 0:
            continue
        samples.append(
            {
                "id": item.get("id"),
                "question": item.get("question"),
                "answer_mode": (item.get("raw_record") or {}).get("answer_mode"),
                "refusal_reason": retrieval_eval.get("refusal_reason"),
                "failure_category": failure_category(item),
                "answer_preview": re.sub(r"\s+", " ", str(item.get("response") or "")).strip()[:180],
            }
        )
    return samples


def failure_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for item in records:
        category = failure_category(item)
        if category == "ok":
            continue
        raw_record = item.get("raw_record") or {}
        failures.append(
            {
                "id": item.get("id"),
                "question": item.get("question"),
                "failure_category": category,
                "route": (item.get("api_response") or {}).get("route"),
                "expected_route": (item.get("retrieval_eval") or {}).get("expected_route"),
                "doc_hit": raw_record.get("doc_hit"),
                "section_hit": raw_record.get("section_hit"),
                "citation_count": raw_record.get("citation_count"),
                "answer_mode": raw_record.get("answer_mode"),
                "refusal_reason": (item.get("retrieval_eval") or {}).get("refusal_reason"),
                "answer_preview": raw_record.get("answer_preview"),
            }
        )
    return failures


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary['label']}",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- route_match_rate: `{summary['route_match_rate']}`",
        f"- doc_id_hit_rate: `{summary['doc_id_hit_rate']}`",
        f"- section_hit_rate: `{summary['section_hit_rate']}`",
        f"- refusal / partial / full: `{summary['mode_counts']['refuse']}` / `{summary['mode_counts']['partial']}` / `{summary['mode_counts']['full']}`",
        f"- citation_count_distribution: `{json.dumps(summary['citation_count_distribution'], ensure_ascii=False)}`",
        f"- support_pack_count_distribution: `{json.dumps(summary['support_pack_count_distribution'], ensure_ascii=False)}`",
        f"- answer_mode_distribution: `{json.dumps(summary['answer_mode_distribution'], ensure_ascii=False)}`",
        f"- v2_validator_flags: `{json.dumps(summary['v2_validator_flags'], ensure_ascii=False)}`",
        "",
        "## Zero Citation",
        "",
    ]
    if summary["zero_citation_samples"]:
        for sample in summary["zero_citation_samples"]:
            lines.append(
                f"- `{sample['id']}` mode=`{sample['answer_mode']}` category=`{sample['failure_category']}` reason=`{sample['refusal_reason']}`"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Failures", ""])
    if summary["failure_samples"]:
        for sample in summary["failure_samples"]:
            lines.append(
                f"- `{sample['id']}` category=`{sample['failure_category']}` route=`{sample['route']}` expected=`{sample['expected_route']}` doc_hit=`{sample['doc_hit']}` section_hit=`{sample['section_hit']}` citations=`{sample['citation_count']}` mode=`{sample['answer_mode']}`"
            )
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    dataset_path = ROOT / "data/eval/datasets/enterprise_ragas_smoke20.json"
    records = load_records(str(dataset_path))
    run_root = ROOT / "reports/evaluation/ad_hoc/generation_v2_stage1_5_compare" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    groups = [
        ("old_neighbor_on", "old + neighbor on"),
        ("old_neighbor_off", "old + neighbor off"),
        ("v2_seed_only", "v2 seed-only"),
    ]
    compare_index: dict[str, Any] = {
        "dataset": str(dataset_path.relative_to(ROOT)),
        "run_root": str(run_root.relative_to(ROOT)),
        "groups": {},
    }

    for group_key, label in groups:
        settings = build_settings(group_key)
        pipeline = SynBioRAGPipeline(settings)
        enriched: list[dict[str, Any]] = []
        for idx, item in enumerate(records, start=1):
            filters = QueryFilters(
                tenant_id=item.get("tenant_id", "default"),
                doc_ids=item.get("doc_ids") or [],
                sections=item.get("sections") or [],
                source_files=item.get("source_files") or [],
                min_score=item.get("min_score"),
            )
            response = pipeline.answer(
                question=item["question"],
                session_id=f"{group_key}_{idx:03d}",
                history=None,
                filters=filters,
            )
            api_response = {
                "session_id": response.session_id,
                "answer": response.answer,
                "confidence": response.confidence,
                "route": response.route.value,
                "citations": [citation.__dict__ for citation in response.citations],
                "used_external_tool": response.used_external_tool,
                "tool_name": response.tool_name,
                "tool_result": response.tool_result,
                "external_references": [ref.__dict__ for ref in response.external_references],
                "debug": response.debug,
            }
            enriched.append(
                {
                    "id": item.get("id") or f"sample_{idx:03d}",
                    "question": item["question"],
                    "reference": (item.get("reference") or "").strip(),
                    "response": api_response.get("answer", ""),
                    "retrieved_contexts": build_contexts(api_response.get("citations") or []),
                    "dataset_meta": {
                        "tags": item.get("tags") or [],
                        "scenario": str(item.get("scenario") or "").strip(),
                        "ability": str(item.get("ability") or "").strip(),
                        "difficulty": str(item.get("difficulty") or "").strip(),
                        "risk_level": str(item.get("risk_level") or "").strip(),
                        "expected_behavior": str(item.get("expected_behavior") or "").strip(),
                        "expected_doc_ids": item.get("expected_doc_ids") or item.get("doc_ids") or [],
                        "accepted_doc_ids": item.get("accepted_doc_ids") or [],
                        "doc_ids": item.get("doc_ids") or [],
                        "expected_source_files": item.get("expected_source_files") or [],
                        "accepted_source_files": item.get("accepted_source_files") or [],
                        "source_files": item.get("source_files") or [],
                        "expected_sections": item.get("expected_sections") or [],
                        "expected_route": str(item.get("expected_route") or "").strip(),
                        "accepted_routes": item.get("accepted_routes") or [],
                        "expected_min_citations": int(item.get("expected_min_citations", 0) or 0),
                        "comparison_branches": item.get("comparison_branches") or [],
                        "expected_min_doc_coverage": normalize_int(item.get("expected_min_doc_coverage")),
                        "allow_partial_if_doc_coverage": normalize_int(item.get("allow_partial_if_doc_coverage")),
                        "allow_partial_answer": bool(item.get("allow_partial_answer")),
                        "expected_answer_mode": str(item.get("expected_answer_mode") or "").strip(),
                        "notes": (item.get("notes") or "").strip(),
                    },
                    "api_response": api_response,
                }
            )

        _, retrieval_summary = evaluate_retrieval(enriched, embeddings=None)
        for item, raw in zip(enriched, build_raw_records(enriched), strict=True):
            item["raw_record"] = raw
            item["retrieval_eval"]["failure_category"] = failure_category(item)
            item["retrieval_eval"]["refusal_reason"] = get_refusal_reason(item)
            item["retrieval_eval"]["final_answer_mode"] = get_effective_final_answer_mode(item)

        diagnostics = build_failure_diagnostics(enriched)
        summary = {
            "group": group_key,
            "label": label,
            "dataset": str(dataset_path.relative_to(ROOT)),
            "sample_count": len(enriched),
            "route_match_rate": retrieval_summary.get("route_match_rate"),
            "doc_id_hit_rate": retrieval_summary.get("doc_id_hit_rate"),
            "section_hit_rate": retrieval_summary.get("section_hit_rate"),
            "citation_count_distribution": citation_distribution(enriched),
            "zero_citation_samples": zero_citation_samples(enriched),
            "mode_counts": dict(Counter(mode_bucket((item.get("raw_record") or {}).get("answer_mode") or "") for item in enriched)),
            "support_pack_count_distribution": support_pack_distribution(enriched),
            "answer_mode_distribution": answer_mode_distribution(enriched),
            "v2_validator_flags": validator_flag_distribution(enriched),
            "failure_category_distribution": dict(sorted(Counter(failure_category(item) for item in enriched if failure_category(item) != "ok").items())),
            "failure_samples": failure_samples(enriched),
            "diagnostics": diagnostics,
            "raw_records": [item.get("raw_record") for item in enriched],
        }
        for key in ("refuse", "partial", "full"):
            summary["mode_counts"].setdefault(key, 0)

        json_path = run_root / f"{group_key}.json"
        md_path = run_root / f"{group_key}.md"
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(build_markdown(summary), encoding="utf-8")
        compare_index["groups"][group_key] = {
            "json": str(json_path.relative_to(ROOT)),
            "md": str(md_path.relative_to(ROOT)),
            "route_match_rate": summary["route_match_rate"],
            "doc_id_hit_rate": summary["doc_id_hit_rate"],
            "section_hit_rate": summary["section_hit_rate"],
            "answer_mode_distribution": summary["answer_mode_distribution"],
        }

    compare_index_path = run_root / "compare_summary.json"
    compare_index_md = run_root / "compare_summary.md"
    compare_index_path.write_text(json.dumps(compare_index, ensure_ascii=False, indent=2), encoding="utf-8")
    compare_lines = ["# Generation V2 Stage 1.5 Compare", ""]
    for group_key, info in compare_index["groups"].items():
        compare_lines.append(
            f"- `{group_key}` route_match_rate=`{info['route_match_rate']}` doc_id_hit_rate=`{info['doc_id_hit_rate']}` section_hit_rate=`{info['section_hit_rate']}` answer_mode_distribution=`{json.dumps(info['answer_mode_distribution'], ensure_ascii=False)}`"
        )
    compare_lines.append("")
    compare_index_md.write_text("\n".join(compare_lines), encoding="utf-8")
    print(json.dumps(compare_index, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
