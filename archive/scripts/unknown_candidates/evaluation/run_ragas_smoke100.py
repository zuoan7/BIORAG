#!/usr/bin/env python3
"""
RAGAS smoke100 自动化测评脚本。

读取 build_ragas_dataset.py 生成的 JSONL，运行 RAGAS 语义指标，
输出 per-sample 分数、汇总统计、分组指标和低分样本列表。

指标:
  - context_recall      检索召回：contexts 覆盖 reference 的程度
  - context_precision   检索精度：contexts 中与 answer 相关的比例
  - faithfulness        忠实度：answer 中声称是否被 contexts 支持
  - answer_relevancy    答案相关性：answer 与 question 的相关程度
  - factual_correctness  事实正确性：answer 与 reference 的事实一致性

使用方式:
  python scripts/evaluation/run_ragas_smoke100.py \\
    --input results/ragas/smoke100_ragas_dataset_final_chunks.jsonl \\
    --output-dir results/ragas/smoke100_<ts> \\
    --metrics context_recall,context_precision,faithfulness,answer_relevancy,factual_correctness

环境变量:
  QWEN_CHAT_API_BASE    Judge LLM API base URL
  QWEN_CHAT_API_KEY     Judge LLM API key
  RAGAS_JUDGE_MODEL     Judge model name (default: qwen-plus)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

warnings.filterwarnings("ignore", message=r"Importing .* from 'ragas.metrics' is deprecated.*",
                        category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r"LangchainLLMWrapper is deprecated.*",
                        category=DeprecationWarning)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Dataset as HFDataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper

# RAGAS 0.4.3 metrics — deprecated import path but functional
from ragas.metrics import (
    context_recall,
    context_precision,
    faithfulness,
    answer_relevancy,
    FactualCorrectness,
)
from ragas.embeddings.base import BaseRagasEmbeddings

from src.synbio_rag.domain.config import Settings
from src.synbio_rag.infrastructure.embedding.bge import BGEM3Embedder


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


# ─── Embedding wrapper ───────────────────────────────────────────
class LocalBGEEmbeddings(BaseRagasEmbeddings):
    def __init__(self, embedder: BGEM3Embedder):
        super().__init__()
        self._embedder = embedder

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.encode([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.encode(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)


# ─── CLI ─────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RAGAS metrics on smoke100 dataset")
    p.add_argument("--input", required=True,
                   help="Path to RAGAS JSONL dataset (from build_ragas_dataset.py)")
    p.add_argument("--output-dir", default="",
                   help="Output directory (default: results/ragas/smoke100_<timestamp>)")
    p.add_argument("--metrics", default="context_recall,context_precision,faithfulness,answer_relevancy,factual_correctness",
                   help="Comma-separated metric names")
    p.add_argument("--judge-model", default="",
                   help="Judge LLM model name (env: RAGAS_JUDGE_MODEL, default: qwen-plus)")
    p.add_argument("--judge-api-base", default="",
                   help="Judge LLM API base URL (env: QWEN_CHAT_API_BASE)")
    p.add_argument("--judge-api-key", default="",
                   help="Judge LLM API key (env: QWEN_CHAT_API_KEY)")
    p.add_argument("--embedding-provider", default="local_bge",
                   choices=["local_bge", "openai"],
                   help="Embedding provider for metrics that need embeddings")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Max samples to evaluate (0 = all)")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="LLM call timeout in seconds")
    p.add_argument("--skip-metrics", default="",
                   help="Comma-separated metrics to skip even if requested")
    return p.parse_args()


# ─── Data loading ────────────────────────────────────────────────
def load_ragas_jsonl(path: str, max_samples: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if max_samples > 0:
        records = records[:max_samples]
    return records


def build_hf_dataset(records: list[dict[str, Any]]) -> HFDataset:
    """Build HuggingFace Dataset for RAGAS evaluate().

    RAGAS 0.4.x expects these column names:
      - user_input (question)
      - response (answer)
      - retrieved_contexts (list[str])
      - reference (ground truth)
    """
    rows: list[dict[str, Any]] = []
    for r in records:
        rows.append({
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": r.get("contexts") or [],
            "reference": r.get("reference") or "",
        })
    return HFDataset.from_list(rows)


# ─── Judge / Embedding setup ─────────────────────────────────────
def build_judge_llm(args: argparse.Namespace) -> LangchainLLMWrapper:
    settings = Settings.from_env()
    api_base = (args.judge_api_base
                or os.getenv("QWEN_CHAT_API_BASE", "")
                or settings.llm.api_base)
    api_key = (args.judge_api_key
               or os.getenv("QWEN_CHAT_API_KEY", "")
               or settings.llm.api_key)
    model = (args.judge_model
             or os.getenv("RAGAS_JUDGE_MODEL", "")
             or "qwen-plus")

    if not api_base or not api_key:
        raise RuntimeError(
            "Missing judge LLM config. Set QWEN_CHAT_API_BASE + QWEN_CHAT_API_KEY env vars, "
            "or pass --judge-api-base / --judge-api-key."
        )

    llm = ChatOpenAI(model=model, base_url=api_base, api_key=api_key,
                     temperature=0.0, timeout=args.timeout)
    return LangchainLLMWrapper(llm)


def build_embeddings(args: argparse.Namespace) -> BaseRagasEmbeddings | None:
    if args.embedding_provider == "local_bge":
        settings = Settings.from_env()
        embedder = BGEM3Embedder(
            model_path=settings.kb.embedding_model_path,
            dim=settings.kb.embedding_dim,
        )
        return LocalBGEEmbeddings(embedder)
    return None


# ─── Metric registry ─────────────────────────────────────────────
METRIC_REGISTRY: dict[str, Any] = {
    "context_recall": context_recall,
    "context_precision": context_precision,
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "factual_correctness": FactualCorrectness(),
}


def resolve_metrics(requested: list[str], skip: list[str]) -> tuple[list[Any], list[str]]:
    """Resolve metric names to metric objects, skipping unavailable ones."""
    metrics: list[Any] = []
    names: list[str] = []
    skipped: list[str] = []
    for name in requested:
        name = name.strip()
        if name in skip:
            skipped.append(f"{name}: skipped_by_request")
            continue
        obj = METRIC_REGISTRY.get(name)
        if obj is None:
            skipped.append(f"{name}: not_in_registry")
            continue
        metrics.append(obj)
        names.append(name)
    return metrics, names, skipped


# ─── Safe evaluation ─────────────────────────────────────────────
def safe_evaluate(dataset: HFDataset, metrics: list[Any],
                  judge_llm: LangchainLLMWrapper,
                  embeddings: BaseRagasEmbeddings | None,
                  metric_names: list[str]) -> tuple[list[dict[str, Any]], dict[str, float], list[str]]:
    """Run RAGAS evaluate with per-metric fallback.

    Returns: (per_sample_scores, summary_dict, failed_metrics)
    """
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=embeddings,
            raise_exceptions=False,
            show_progress=True,
        )
    except Exception as exc:
        print(f"[ERROR] Batch evaluation failed: {exc}")
        return [], {}, metric_names

    df = result.to_pandas()
    # Normalize column names
    col_map: dict[str, str] = {}
    for col in df.columns:
        if col == "user_input":
            continue
        normalized = re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_")
        col_map[col] = normalized

    per_sample: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        score_row: dict[str, Any] = {}
        for orig, norm in col_map.items():
            val = row[orig]
            if isinstance(val, float) and math.isnan(val):
                val = None
            score_row[norm] = val
        per_sample.append(score_row)

    # Summary
    summary: dict[str, float] = {}
    for norm in set(col_map.values()):
        values = [s[norm] for s in per_sample if isinstance(s.get(norm), (int, float))]
        if values:
            summary[norm] = round(mean(values), 4)

    return per_sample, summary, []


# ─── Merge scores with metadata ──────────────────────────────────
def merge_with_metadata(per_sample: list[dict[str, Any]],
                        records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge RAGAS scores back into original records."""
    merged: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        entry = dict(rec)
        if i < len(per_sample):
            entry["ragas_scores"] = per_sample[i]
        else:
            entry["ragas_scores"] = {}
        merged.append(entry)
    return merged


# ─── Group statistics ────────────────────────────────────────────
def mean_or_none(values: list[float]) -> float | None:
    numeric = [v for v in values if isinstance(v, (int, float)) and not _is_nan(v)]
    return round(mean(numeric), 4) if numeric else None


def compute_group_stats(merged: list[dict[str, Any]],
                        metric_names: list[str]) -> dict[str, Any]:
    """Compute group-level statistics for route, scenario, expected_behavior, etc."""
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "route": {},
        "scenario": {},
        "expected_behavior": {},
        "doc_id_hit": {"true": [], "false": []},
        "section_norm_hit": {"true": [], "false": []},
        "citation_count_zero": {"true": [], "false": []},
        "empty_context": {"true": [], "false": []},
        "answer_mode": {},
    }

    for item in merged:
        route = item.get("route", "unknown")
        scenario = item.get("scenario", "unknown")
        behavior = item.get("expected_behavior", "unknown")
        answer_mode = item.get("answer_mode", "unknown")

        groups["route"].setdefault(route, []).append(item)
        groups["scenario"].setdefault(scenario, []).append(item)
        groups["expected_behavior"].setdefault(behavior, []).append(item)
        groups["doc_id_hit"]["true" if item.get("doc_id_hit") else "false"].append(item)
        groups["section_norm_hit"]["true" if item.get("section_norm_hit") else "false"].append(item)
        groups["citation_count_zero"]["true" if item.get("citation_count", 0) == 0 else "false"].append(item)
        groups["empty_context"]["true" if item.get("empty_context") else "false"].append(item)
        groups["answer_mode"].setdefault(answer_mode, []).append(item)

    result: dict[str, Any] = {}
    for dim, buckets in groups.items():
        dim_result: dict[str, Any] = {}
        for bucket, items in sorted(buckets.items()):
            if not items:
                continue
            stats: dict[str, Any] = {"count": len(items)}
            for m in metric_names:
                values = [(it.get("ragas_scores") or {}).get(m) for it in items]
                avg = mean_or_none(values)
                if avg is not None:
                    stats[m] = avg
            dim_result[bucket] = stats
        if dim_result:
            result[dim] = dim_result

    return result


# ─── Low-score samples ───────────────────────────────────────────
def suspected_issue_type(item: dict[str, Any], metric: str, score: float) -> str:
    """Heuristic classification of suspected issue for a low score."""
    empty_ctx = item.get("empty_context", False)
    answer_mode = item.get("answer_mode", "")
    doc_hit = item.get("doc_id_hit", True)
    section_hit = item.get("section_norm_hit", True)
    citation_count = item.get("citation_count", 0)

    if answer_mode in ("empty", "error"):
        return "empty_context"
    if answer_mode == "refusal":
        return "refusal_case"
    if empty_ctx:
        return "empty_context"
    if not doc_hit and metric in ("context_recall", "context_precision"):
        return "retrieval_missing"
    if metric == "faithfulness" and score < 0.5:
        return "answer_not_grounded"
    if metric == "factual_correctness" and score < 0.5:
        if doc_hit:
            return "possible_reference_mismatch"
        return "retrieval_missing"
    if metric == "answer_relevancy" and score < 0.5:
        return "irrelevant_context"
    if metric in ("context_recall", "context_precision") and score < 0.5:
        if section_hit and citation_count > 0:
            return "irrelevant_context"
        return "retrieval_missing"
    if score < 0.3:
        return "judge_uncertain"
    return "answer_incomplete"


def build_low_score_lists(merged: list[dict[str, Any]],
                          metric_names: list[str],
                          top_n: int = 10) -> dict[str, list[dict[str, Any]]]:
    """Build top-N low-score lists per metric."""
    low_score: dict[str, list[dict[str, Any]]] = {}

    for m in metric_names:
        # Sort by this metric ascending (None → lowest)
        scored = []
        for item in merged:
            score = (item.get("ragas_scores") or {}).get(m)
            if score is None:
                score_val = -1.0  # push None to top
            else:
                score_val = float(score)
            scored.append((score_val, item))

        scored.sort(key=lambda x: x[0])
        bottom = []
        for score_val, item in scored[:top_n]:
            scores = item.get("ragas_scores") or {}
            bottom.append({
                "sample_id": item.get("sample_id", ""),
                "question": item.get("question", ""),
                "answer_preview": _truncate(item.get("answer", ""), 200),
                "reference_preview": _truncate(item.get("reference", ""), 200),
                "contexts_preview": _truncate(
                    " | ".join((item.get("contexts") or [])[:3]), 300
                ),
                "route": item.get("route", ""),
                "scenario": item.get("scenario", ""),
                "expected_behavior": item.get("expected_behavior", ""),
                "doc_id_hit": item.get("doc_id_hit", False),
                "section_norm_hit": item.get("section_norm_hit", False),
                "citation_count": item.get("citation_count", 0),
                "metric_score": scores.get(m),
                "suspected_issue_type": suspected_issue_type(item, m, float(scores.get(m) or 0)),
            })
        low_score[f"{m}_bottom{top_n}"] = bottom

    return low_score


def _truncate(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text[:limit] + ("..." if len(text) > limit else "")


# ─── Report generation ───────────────────────────────────────────
def build_summary_md(global_stats: dict[str, float],
                     group_stats: dict[str, Any],
                     metric_names: list[str],
                     context_source: str,
                     skipped: list[str]) -> str:
    lines = [
        "# RAGAS Smoke100 Evaluation Summary",
        "",
        f"**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Context source**: `{context_source}`",
        f"**Metrics evaluated**: {', '.join(metric_names)}",
        "",
    ]
    if skipped:
        lines += ["## Skipped Metrics", ""]
        for s in skipped:
            lines.append(f"- `{s}`")
        lines.append("")

    # Global averages
    lines += ["## Global Averages", "", "| Metric | Score |", "|--------|-------|"]
    for m in metric_names:
        val = global_stats.get(m)
        display = f"`{val}`" if val is not None else "N/A"
        lines.append(f"| {m} | {display} |")
    lines.append("")

    # Group statistics
    for dim in ["route", "scenario", "expected_behavior"]:
        dim_data = group_stats.get(dim, {})
        if not dim_data:
            continue
        lines += [f"## By {dim.replace('_', ' ').title()}", ""]
        header = "| Group | Count | " + " | ".join(metric_names) + " |"
        lines.append(header)
        sep = "|-------|-------|" + "|".join(["-------"] * len(metric_names)) + "|"
        lines.append(sep)
        for bucket, stats in sorted(dim_data.items()):
            row = f"| {bucket} | {stats.get('count', '-')} |"
            for m in metric_names:
                val = stats.get(m)
                row += f" `{val}` |" if val is not None else " - |"
            lines.append(row)
        lines.append("")

    # Project metrics cross-tabs
    for dim in ["doc_id_hit", "section_norm_hit", "citation_count_zero", "empty_context", "answer_mode"]:
        dim_data = group_stats.get(dim, {})
        if not dim_data:
            continue
        lines += [f"## By {dim}", ""]
        header = "| Group | Count | " + " | ".join(metric_names) + " |"
        lines.append(header)
        sep = "|-------|-------|" + "|".join(["-------"] * len(metric_names)) + "|"
        lines.append(sep)
        for bucket, stats in sorted(dim_data.items()):
            row = f"| {bucket} | {stats.get('count', '-')} |"
            for m in metric_names:
                val = stats.get(m)
                row += f" `{val}` |" if val is not None else " - |"
            lines.append(row)
        lines.append("")

    return "\n".join(lines)


def build_low_score_md(low_score_lists: dict[str, list[dict[str, Any]]],
                       metric_names: list[str]) -> str:
    lines = ["# RAGAS Low-Score Cases", ""]

    for m in metric_names:
        key = f"{m}_bottom10"
        cases = low_score_lists.get(key, [])
        lines += [f"## {m} — Bottom 10", ""]
        if not cases:
            lines.append("No cases.\n")
            continue
        for rank, case in enumerate(cases, 1):
            lines += [
                f"### {rank}. `{case['sample_id']}` — score: `{case['metric_score']}`",
                "",
                f"- **Question**: {case['question']}",
                f"- **Answer**: {case['answer_preview']}",
                f"- **Reference**: {case['reference_preview']}",
                f"- **Contexts**: {case['contexts_preview']}",
                f"- **Route**: `{case['route']}` | **Scenario**: `{case['scenario']}` | **Behavior**: `{case['expected_behavior']}`",
                f"- **doc_id_hit**: `{case['doc_id_hit']}` | **section_norm_hit**: `{case['section_norm_hit']}` | **citations**: `{case['citation_count']}`",
                f"- **Suspected issue**: `{case['suspected_issue_type']}`",
                "",
            ]
    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────
def main() -> int:
    args = parse_args()

    # Setup output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "results/ragas" / f"smoke100_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Context source from input filename
    input_path = Path(args.input)
    context_source = "unknown"
    if "cited_chunks" in input_path.stem:
        context_source = "cited_chunks"
    elif "final_chunks" in input_path.stem:
        context_source = "final_chunks"

    # Load data
    records = load_ragas_jsonl(args.input, args.max_samples)
    print(f"[run_ragas] Loaded {len(records)} samples from {args.input}")

    # Resolve metrics
    requested_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    skip_metrics = [m.strip() for m in args.skip_metrics.split(",") if m.strip()] if args.skip_metrics else []
    metrics, metric_names, skipped = resolve_metrics(requested_metrics, skip_metrics)
    print(f"[run_ragas] Metrics: {metric_names}")
    if skipped:
        print(f"[run_ragas] Skipped: {skipped}")

    if not metrics:
        print("[run_ragas] No metrics to evaluate. Exiting.")
        return 1

    # Build HF dataset
    hf_dataset = build_hf_dataset(records)

    # Judge LLM
    try:
        judge_llm = build_judge_llm(args)
        print(f"[run_ragas] Judge LLM configured")
    except RuntimeError as exc:
        print(f"[run_ragas] SKIP: Cannot configure judge LLM: {exc}")
        # Write a placeholder report
        _write_blocked_report(output_dir, metric_names, context_source, str(exc))
        return 1

    # Embeddings
    embeddings = build_embeddings(args)

    # Evaluate
    per_sample, global_stats, failed = safe_evaluate(
        hf_dataset, metrics, judge_llm, embeddings, metric_names
    )
    if failed:
        print(f"[run_ragas] Failed metrics: {failed}")

    # Merge
    merged = merge_with_metadata(per_sample, records)

    # Group stats
    group_stats = compute_group_stats(merged, metric_names)

    # Low-score lists
    low_score_lists = build_low_score_lists(merged, metric_names, top_n=10)

    # Write outputs
    # 1. Per-sample scores JSONL
    scores_path = output_dir / "ragas_scores.jsonl"
    with scores_path.open("w", encoding="utf-8") as fh:
        for item in merged:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 2. Summary JSON
    summary_path = output_dir / "ragas_summary.json"
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "context_source": context_source,
        "sample_count": len(records),
        "metrics_evaluated": metric_names,
        "skipped": skipped,
        "global_averages": global_stats,
        "group_statistics": group_stats,
    }
    summary_path.write_text(json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3. Summary MD
    md_path = output_dir / "ragas_summary.md"
    md_path.write_text(build_summary_md(global_stats, group_stats, metric_names, context_source, skipped),
                       encoding="utf-8")

    # 4. Low-score cases MD
    low_md_path = output_dir / "ragas_low_score_cases.md"
    low_md_path.write_text(build_low_score_md(low_score_lists, metric_names), encoding="utf-8")

    # Print summary
    print("\n[Global Averages]")
    for m in metric_names:
        val = global_stats.get(m)
        print(f"  {m}: {val}" if val is not None else f"  {m}: N/A")

    print(f"\n[Outputs]")
    print(f"  Scores:    {scores_path}")
    print(f"  Summary:   {summary_path}")
    print(f"  Report:    {md_path}")
    print(f"  Low-score: {low_md_path}")

    return 0


def _write_blocked_report(output_dir: Path, metric_names: list[str],
                          context_source: str, reason: str) -> None:
    """Write a report explaining why evaluation could not run."""
    lines = [
        "# RAGAS Smoke100 — Blocked",
        "",
        f"**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Status: Cannot Run",
        "",
        f"**Reason**: {reason}",
        "",
        "## Required Configuration",
        "",
        "Set the following environment variables:",
        "```bash",
        "export QWEN_CHAT_API_BASE=<your-api-base-url>",
        "export QWEN_CHAT_API_KEY=<your-api-key>",
        "export RAGAS_JUDGE_MODEL=qwen-plus   # optional",
        "```",
        "",
        "Or pass them as CLI arguments:",
        "```bash",
        "python scripts/evaluation/run_ragas_smoke100.py \\",
        "  --input results/ragas/smoke100_ragas_dataset_final_chunks.jsonl \\",
        "  --judge-api-base <url> \\",
        "  --judge-api-key <key> \\",
        "  --judge-model qwen-plus",
        "```",
    ]
    md_path = output_dir / "ragas_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    blocked_data = {
        "status": "blocked",
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
        "metrics_requested": metric_names,
        "context_source": context_source,
    }
    json_path = output_dir / "ragas_summary.json"
    json_path.write_text(json.dumps(blocked_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Blocked] Report written to {output_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
