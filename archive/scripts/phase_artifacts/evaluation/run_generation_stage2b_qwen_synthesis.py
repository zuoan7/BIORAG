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

FOCUS_SAMPLE_IDS = [
    "ent_094",
    "ent_007",
    "ent_020",
    "ent_021",
    "ent_092",
    "ent_015",
    "ent_026",
    "ent_064",
]


def normalize_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_settings(group_key: str) -> Settings:
    settings = Settings.from_env()
    settings.audit.enabled = False
    settings.generation.version = "v2"
    settings.generation.v2_use_external_tools = False
    settings.generation.v2_use_history = False
    settings.retrieval.neighbor_expansion_enabled = True
    settings.generation.v2_use_qwen_synthesis = group_key == "v2_qwen_synthesis"
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


def answer_mode_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in records:
        counter[str((item.get("raw_record") or {}).get("answer_mode") or "unknown")] += 1
    return dict(sorted(counter.items()))


def qwen_fallback_reason_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in records:
        qwen_debug = ((item.get("raw_record") or {}).get("debug") or {}).get("generation_v2", {}).get("qwen_synthesis") or {}
        reason = str(qwen_debug.get("fallback_reason") or "")
        if reason:
            counter[reason] += 1
    return dict(sorted(counter.items()))


def qwen_validation_flag_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in records:
        qwen_debug = ((item.get("raw_record") or {}).get("debug") or {}).get("generation_v2", {}).get("qwen_synthesis") or {}
        for flag in qwen_debug.get("validation_flags") or []:
            counter[str(flag)] += 1
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
        qwen_debug = (raw_record.get("debug") or {}).get("generation_v2", {}).get("qwen_synthesis") or {}
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
                "qwen_fallback_reason": qwen_debug.get("fallback_reason"),
                "qwen_validation_flags": qwen_debug.get("validation_flags") or [],
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
        f"- answer_mode_distribution: `{json.dumps(summary['answer_mode_distribution'], ensure_ascii=False)}`",
        f"- qwen_used_count: `{summary['qwen_used_count']}`",
        f"- qwen_fallback_count: `{summary['qwen_fallback_count']}`",
        f"- qwen_fallback_reason_distribution: `{json.dumps(summary['qwen_fallback_reason_distribution'], ensure_ascii=False)}`",
        f"- qwen_validation_flag_distribution: `{json.dumps(summary['qwen_validation_flag_distribution'], ensure_ascii=False)}`",
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
                f"- `{sample['id']}` category=`{sample['failure_category']}` mode=`{sample['answer_mode']}` citations=`{sample['citation_count']}` qwen_fallback=`{sample['qwen_fallback_reason']}` flags=`{sample['qwen_validation_flags']}`"
            )
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def run_group(group_key: str, label: str, dataset_path: Path, records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        "answer_mode_distribution": answer_mode_distribution(enriched),
        "qwen_used_count": sum(
            1
            for item in enriched
            if (((item.get("raw_record") or {}).get("debug") or {}).get("generation_v2", {}).get("qwen_synthesis") or {}).get("used_qwen")
        ),
        "qwen_fallback_count": sum(
            1
            for item in enriched
            if (((item.get("raw_record") or {}).get("debug") or {}).get("generation_v2", {}).get("qwen_synthesis") or {}).get("fallback_used")
        ),
        "qwen_fallback_reason_distribution": qwen_fallback_reason_distribution(enriched),
        "qwen_validation_flag_distribution": qwen_validation_flag_distribution(enriched),
        "failure_category_distribution": dict(sorted(Counter(failure_category(item) for item in enriched if failure_category(item) != "ok").items())),
        "failure_samples": failure_samples(enriched),
        "diagnostics": diagnostics,
        "raw_records": [item.get("raw_record") for item in enriched],
    }
    for key in ("refuse", "partial", "full"):
        summary["mode_counts"].setdefault(key, 0)
    return summary, enriched


def build_focus_samples(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for sample_id in FOCUS_SAMPLE_IDS:
        payload[sample_id] = {}
        for group_key, records in groups.items():
            for item in records:
                if item.get("id") != sample_id:
                    continue
                raw = item.get("raw_record") or {}
                payload[sample_id][group_key] = {
                    "id": sample_id,
                    "question": item.get("question"),
                    "answer_mode": raw.get("answer_mode"),
                    "citation_count": raw.get("citation_count"),
                    "doc_hit": raw.get("doc_hit"),
                    "section_hit": raw.get("section_hit"),
                    "failure_category": failure_category(item),
                    "answer_preview": raw.get("answer_preview"),
                    "qwen_synthesis": (raw.get("debug") or {}).get("generation_v2", {}).get("qwen_synthesis") or {},
                    "existence_guardrail": (raw.get("debug") or {}).get("generation_v2", {}).get("existence_guardrail") or {},
                }
                break
    return payload


def build_diff_vs_stage2a(stage2a_summary: dict[str, Any], qwen_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_stage2a_group": stage2a_summary.get("group"),
        "current_group": qwen_summary.get("group"),
        "metrics": {
            "route_match_rate": {
                "stage2a": stage2a_summary.get("route_match_rate"),
                "current": qwen_summary.get("route_match_rate"),
            },
            "doc_id_hit_rate": {
                "stage2a": stage2a_summary.get("doc_id_hit_rate"),
                "current": qwen_summary.get("doc_id_hit_rate"),
            },
            "section_hit_rate": {
                "stage2a": stage2a_summary.get("section_hit_rate"),
                "current": qwen_summary.get("section_hit_rate"),
            },
            "answer_mode_distribution": {
                "stage2a": stage2a_summary.get("answer_mode_distribution"),
                "current": qwen_summary.get("answer_mode_distribution"),
            },
            "citation_count_distribution": {
                "stage2a": stage2a_summary.get("citation_count_distribution"),
                "current": qwen_summary.get("citation_count_distribution"),
            },
        },
        "qwen_used_count": qwen_summary.get("qwen_used_count"),
        "qwen_fallback_count": qwen_summary.get("qwen_fallback_count"),
        "qwen_fallback_reason_distribution": qwen_summary.get("qwen_fallback_reason_distribution"),
    }


def main() -> int:
    dataset_path = ROOT / "data/eval/datasets/enterprise_ragas_smoke20.json"
    records = load_records(str(dataset_path))
    run_root = ROOT / "reports/evaluation/ad_hoc/generation_v2_stage2b_qwen_synthesis" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    groups = [
        ("v2_extractive_baseline", "v2 extractive baseline"),
        ("v2_qwen_synthesis", "v2 qwen synthesis"),
    ]
    compare_index: dict[str, Any] = {
        "dataset": str(dataset_path.relative_to(ROOT)),
        "run_root": str(run_root.relative_to(ROOT)),
        "groups": {},
    }
    group_records: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}

    for group_key, label in groups:
        summary, enriched = run_group(group_key, label, dataset_path, records)
        summaries[group_key] = summary
        group_records[group_key] = enriched
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
            "qwen_used_count": summary["qwen_used_count"],
            "qwen_fallback_count": summary["qwen_fallback_count"],
        }

    stage2a_path = ROOT / "reports/evaluation/ad_hoc/generation_v2_stage2a_existence_guardrail/20260427_124804/v2_seed_only.json"
    stage2a_summary = json.loads(stage2a_path.read_text(encoding="utf-8"))
    diff_payload = build_diff_vs_stage2a(stage2a_summary, summaries["v2_qwen_synthesis"])
    diff_json = run_root / "diff_vs_stage2a.json"
    diff_md = run_root / "diff_vs_stage2a.md"
    diff_json.write_text(json.dumps(diff_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    diff_md.write_text(
        "\n".join(
            [
                "# Diff Vs Stage 2A",
                "",
                f"- route_match_rate: `{diff_payload['metrics']['route_match_rate']['stage2a']}` -> `{diff_payload['metrics']['route_match_rate']['current']}`",
                f"- doc_id_hit_rate: `{diff_payload['metrics']['doc_id_hit_rate']['stage2a']}` -> `{diff_payload['metrics']['doc_id_hit_rate']['current']}`",
                f"- section_hit_rate: `{diff_payload['metrics']['section_hit_rate']['stage2a']}` -> `{diff_payload['metrics']['section_hit_rate']['current']}`",
                f"- answer_mode_distribution: `{json.dumps(diff_payload['metrics']['answer_mode_distribution']['stage2a'], ensure_ascii=False)}` -> `{json.dumps(diff_payload['metrics']['answer_mode_distribution']['current'], ensure_ascii=False)}`",
                f"- citation_count_distribution: `{json.dumps(diff_payload['metrics']['citation_count_distribution']['stage2a'], ensure_ascii=False)}` -> `{json.dumps(diff_payload['metrics']['citation_count_distribution']['current'], ensure_ascii=False)}`",
                f"- qwen_used_count: `{diff_payload['qwen_used_count']}`",
                f"- qwen_fallback_count: `{diff_payload['qwen_fallback_count']}`",
                f"- qwen_fallback_reason_distribution: `{json.dumps(diff_payload['qwen_fallback_reason_distribution'], ensure_ascii=False)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    focus_payload = build_focus_samples(group_records)
    (run_root / "focus_samples.json").write_text(json.dumps(focus_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    compare_summary_path = run_root / "compare_summary.json"
    compare_summary_md = run_root / "compare_summary.md"
    compare_summary_path.write_text(json.dumps(compare_index, ensure_ascii=False, indent=2), encoding="utf-8")
    compare_summary_md.write_text(
        "\n".join(
            [
                "# Generation V2 Stage 2B Compare",
                "",
                *[
                    f"- `{group_key}` route_match_rate=`{info['route_match_rate']}` doc_id_hit_rate=`{info['doc_id_hit_rate']}` section_hit_rate=`{info['section_hit_rate']}` answer_mode_distribution=`{json.dumps(info['answer_mode_distribution'], ensure_ascii=False)}` qwen_used_count=`{info['qwen_used_count']}` qwen_fallback_count=`{info['qwen_fallback_count']}`"
                    for group_key, info in compare_index["groups"].items()
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(compare_index, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
