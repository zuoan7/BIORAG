#!/usr/bin/env python3
"""
主要功能:
- 调用当前 RAG 服务的 /v1/ask 接口，批量生成评测答案与引用上下文。
- 使用 ragas 对回答相关性、忠实度，以及可选的参考答案一致性做自动化评估。
- 输出按检索/生成拆分的评测结果，并可把逐条结果写入 JSON 文件，便于回归比较。

启动方法:
- 先启动 RAG 服务，再准备一个 JSONL/JSON 评测集文件。
- 基本用法:
  source /home/gmy/miniconda3/bin/activate bge
  python scripts/evaluation/evaluate_ragas.py --dataset data/eval/datasets/ragas_eval_sample.jsonl

- 如需指定评审模型:
  python scripts/evaluation/evaluate_ragas.py \
    --dataset data/eval/datasets/ragas_eval_sample.jsonl \
    --judge-model qwen-plus
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
import math
import os
import re
import sys
import warnings
from pathlib import Path
from statistics import mean
from typing import Any


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)

import httpx
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import LangchainLLMWrapper

warnings.filterwarnings(
    "ignore",
    message=r"Importing .* from 'ragas.metrics' is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"LangchainLLMWrapper is deprecated.*",
    category=DeprecationWarning,
)
from ragas.metrics import answer_correctness, answer_relevancy, context_recall, faithfulness

ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag.domain.config import Settings
from src.synbio_rag.infrastructure.embedding.bge import BGEM3Embedder


class LocalBGEEmbeddings(BaseRagasEmbeddings):
    def __init__(self, embedder: BGEM3Embedder):
        super().__init__()
        self.embedder = embedder

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.encode([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.encode(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 ragas 自动评估当前 SynBio RAG 服务")
    parser.add_argument(
        "--dataset",
        default="data/eval/datasets/enterprise_ragas_eval_v1.json",
        help="评测集路径，支持 .jsonl / .json，默认: data/eval/datasets/enterprise_ragas_eval_v1.json",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="RAG API 地址，默认: http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--judge-model",
        default="qwen-plus",
        help="用于 ragas 裁判的模型名，默认: qwen-plus",
    )
    parser.add_argument(
        "--judge-api-base",
        default="",
        help="裁判模型 API Base，默认从 .env 或环境变量 QWEN_CHAT_API_BASE 读取",
    )
    parser.add_argument(
        "--judge-api-key",
        default="",
        help="裁判模型 API Key，默认从 .env 或环境变量 QWEN_CHAT_API_KEY 读取",
    )
    parser.add_argument(
        "--output",
        default="",
        help="评测报告输出路径；未指定时自动写入 reports/evaluation/ragas/runs/",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP 请求超时秒数，默认 180",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="最多评测多少条，0 表示全部",
    )
    parser.add_argument(
        "--include-debug",
        action="store_true",
        help="请求 /v1/ask 时带 include_debug=true",
    )
    parser.add_argument(
        "--skip-reference-metrics",
        action="store_true",
        help="跳过需要 reference 的指标，只跑 answer_relevancy 和 faithfulness",
    )
    parser.add_argument(
        "--skip-generation-metrics",
        action="store_true",
        help="跳过 ragas 生成指标，只输出检索报告",
    )
    parser.add_argument(
        "--enable-round8-policy",
        action="store_true",
        help="记录 Round8 policy 评测开关；服务端需同步设置 ROUND8_ENABLE_ROUND8_POLICY=true",
    )
    parser.add_argument(
        "--disable-comparison-single-doc-hard-refusal",
        action="store_true",
        help=(
            "记录 Round8.1 单文档 comparison 硬拒禁用开关；服务端需同步设置 "
            "ROUND8_DISABLE_COMPARISON_SINGLE_DOC_HARD_REFUSAL=true"
        ),
    )
    parser.add_argument(
        "--enable-claim-fallback",
        action="store_true",
        help="记录 Round8.2 claim fallback 开关；服务端需同步设置 ROUND8_ENABLE_CLAIM_FALLBACK=true",
    )
    parser.add_argument(
        "--enable-partial-answer",
        action="store_true",
        help="记录 Round8.2 partial answer 开关；服务端需同步设置 ROUND8_ENABLE_PARTIAL_ANSWER=true",
    )
    parser.add_argument(
        "--enable-route-specific-thresholds",
        action="store_true",
        help="记录 Round8.3 route-specific threshold 开关；服务端需同步设置 ROUND8_ENABLE_ROUTE_SPECIFIC_THRESHOLDS=true",
    )
    parser.add_argument(
        "--enable-comparison-prompt-v2",
        action="store_true",
        help="记录 Round8.3 comparison prompt v2 开关；服务端需同步设置 ROUND8_ENABLE_COMPARISON_PROMPT_V2=true",
    )
    return parser.parse_args()


def build_default_output_path(dataset_path: str) -> Path:
    dataset_stem = Path(dataset_path).stem or "dataset"
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", dataset_stem).strip("_") or "dataset"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RESULTS_ROOT / "ad_hoc" / safe_stem / timestamp / "report.json"


def resolve_output_path(raw_output: str, dataset_path: str) -> Path:
    output_path = Path(raw_output) if raw_output else build_default_output_path(dataset_path)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    if output_path.exists() and output_path.is_dir():
        return output_path / "report.json"
    if output_path.suffix.lower() == ".json":
        return output_path
    return output_path / "report.json"


def build_run_config(args: argparse.Namespace, output_path: Path, base_url: str) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "output": str(output_path.relative_to(ROOT)) if output_path.is_relative_to(ROOT) else str(output_path),
        "base_url": base_url,
        "judge_model": args.judge_model,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "flags": {
            "include_debug": args.include_debug,
            "skip_reference_metrics": args.skip_reference_metrics,
            "skip_generation_metrics": args.skip_generation_metrics,
            "enable_round8_policy": args.enable_round8_policy,
            "disable_comparison_single_doc_hard_refusal": args.disable_comparison_single_doc_hard_refusal,
            "enable_claim_fallback": args.enable_claim_fallback,
            "enable_partial_answer": args.enable_partial_answer,
            "enable_route_specific_thresholds": args.enable_route_specific_thresholds,
            "enable_comparison_prompt_v2": args.enable_comparison_prompt_v2,
            "max_samples": args.max_samples,
            "timeout": args.timeout,
        },
    }


def describe_result_file(path: Path) -> str:
    name = path.name
    if name == "report.json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return "structured evaluation report"
        dataset = payload.get("dataset") or "unknown_dataset"
        sample_count = payload.get("sample_count")
        report_type = payload.get("report_type") or "report"
        return f"{report_type}; dataset={dataset}; sample_count={sample_count}"
    if name == "run_config.json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return "run configuration metadata"
        return f"run metadata; dataset={payload.get('dataset')}; base_url={payload.get('base_url')}"
    if name.endswith(".md"):
        return "markdown summary"
    if name.endswith(".json"):
        stem = path.stem
        if "summary" in stem:
            return "aggregated summary"
        return "legacy flat evaluation report"
    return "result artifact"


def refresh_results_index() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path for path in RESULTS_ROOT.rglob("*")
        if path.is_file() and path.name != "INDEX.md"
    )

    lines = ["# Results Index", "", "## Structured Results", ""]
    structured = [path for path in files if path.name in {"report.json", "run_config.json"}]
    if structured:
        lines.append("| Path | Description |")
        lines.append("|---|---|")
        for path in structured:
            rel = path.relative_to(ROOT)
            lines.append(f"| `{rel.as_posix()}` | {describe_result_file(path)} |")
    else:
        lines.append("None.")

    lines.extend(["", "## Summary And Legacy Files", ""])
    legacy = [path for path in files if path.name not in {"report.json", "run_config.json"}]
    if legacy:
        lines.append("| Path | Description |")
        lines.append("|---|---|")
        for path in legacy:
            rel = path.relative_to(ROOT)
            lines.append(f"| `{rel.as_posix()}` | {describe_result_file(path)} |")
    else:
        lines.append("None.")

    (RESULTS_ROOT / "INDEX.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def load_records(dataset_path: str) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"评测集不存在: {path}")
    if path.suffix.lower() == ".jsonl":
        items = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    items.append(json.loads(raw))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL 第 {line_no} 行解析失败: {exc}") from exc
        return items
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("samples", "items", "data"):
                values = payload.get(key)
                if isinstance(values, list):
                    return values
        raise ValueError("JSON 评测集必须是对象数组，或包含 samples/items/data 数组")
    raise ValueError(f"不支持的评测集格式: {path.suffix}")


def normalize_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def normalize_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def slugify_metric_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def build_payload(item: dict[str, Any], include_debug: bool) -> dict[str, Any]:
    question = (item.get("question") or "").strip()
    if not question:
        raise ValueError("评测集中的 question 不能为空")
    return {
        "question": question,
        "session_id": item.get("session_id"),
        "tenant_id": item.get("tenant_id", "default"),
        "doc_ids": item.get("doc_ids") or [],
        "sections": item.get("sections") or [],
        "source_files": item.get("source_files") or [],
        "min_score": item.get("min_score"),
        "include_debug": include_debug,
    }


def fetch_answers(
    base_url: str,
    records: list[dict[str, Any]],
    timeout: float,
    include_debug: bool,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    with httpx.Client(timeout=timeout) as client:
        healthz = client.get(f"{base_url}/healthz")
        healthz.raise_for_status()
        print(json.dumps({"healthz": healthz.json()}, ensure_ascii=False, indent=2))

        for idx, item in enumerate(records, start=1):
            payload = build_payload(item, include_debug)
            response = client.post(f"{base_url}/v1/ask", json=payload)
            response.raise_for_status()
            result = response.json()
            contexts = []
            for citation in result.get("citations") or []:
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

            enriched.append(
                {
                    "id": item.get("id") or f"sample_{idx:03d}",
                    "question": item["question"],
                    "reference": (item.get("reference") or "").strip(),
                    "response": result.get("answer", ""),
                    "retrieved_contexts": contexts,
                    "dataset_meta": {
                        "tags": normalize_list(item.get("tags")),
                        "scenario": str(item.get("scenario") or "").strip(),
                        "ability": str(item.get("ability") or "").strip(),
                        "difficulty": str(item.get("difficulty") or "").strip(),
                        "risk_level": str(item.get("risk_level") or "").strip(),
                        "expected_behavior": str(item.get("expected_behavior") or "").strip(),
                        "expected_doc_ids": normalize_list(item.get("expected_doc_ids") or item.get("doc_ids")),
                        "accepted_doc_ids": normalize_list(item.get("accepted_doc_ids")),
                        "doc_ids": normalize_list(item.get("doc_ids")),
                        "expected_source_files": normalize_list(item.get("expected_source_files")),
                        "accepted_source_files": normalize_list(item.get("accepted_source_files")),
                        "source_files": normalize_list(item.get("source_files")),
                        "expected_sections": normalize_list(item.get("expected_sections")),
                        "expected_route": (item.get("expected_route") or "").strip(),
                        "accepted_routes": normalize_list(item.get("accepted_routes")),
                        "expected_min_citations": int(item.get("expected_min_citations", 0) or 0),
                        "comparison_branches": item.get("comparison_branches") or [],
                        "expected_min_doc_coverage": normalize_int(item.get("expected_min_doc_coverage")),
                        "allow_partial_if_doc_coverage": normalize_int(item.get("allow_partial_if_doc_coverage")),
                        "allow_partial_answer": bool(item.get("allow_partial_answer")),
                        "expected_answer_mode": str(item.get("expected_answer_mode") or "").strip(),
                        "notes": (item.get("notes") or "").strip(),
                    },
                    "api_response": result,
                }
            )
            print(
                json.dumps(
                    {
                        "sample": enriched[-1]["id"],
                        "question": item["question"],
                        "citations": len(contexts),
                        "confidence": result.get("confidence"),
                        "route": result.get("route"),
                    },
                    ensure_ascii=False,
                )
            )
    return enriched


def build_ragas_dataset(records: list[dict[str, Any]]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "user_input": item["question"],
                "response": item["response"],
                "retrieved_contexts": item["retrieved_contexts"],
                "reference": item["reference"],
            }
            for item in records
        ]
    )


def build_metrics(has_reference: bool, skip_reference_metrics: bool) -> list[Any]:
    metrics: list[Any] = [answer_relevancy, faithfulness]
    if has_reference and not skip_reference_metrics:
        metrics.extend([context_recall, answer_correctness])
    return metrics


def build_judge_llm(args: argparse.Namespace) -> LangchainLLMWrapper:
    settings = Settings.from_env()
    api_base = args.judge_api_base or settings.llm.api_base or os.getenv("QWEN_CHAT_API_BASE", "")
    api_key = args.judge_api_key or settings.llm.api_key or os.getenv("QWEN_CHAT_API_KEY", "")
    if not api_base or not api_key:
        raise RuntimeError(
            "缺少裁判模型配置。请设置 --judge-api-base/--judge-api-key，或在 .env 中提供 "
            "QWEN_CHAT_API_BASE 与 QWEN_CHAT_API_KEY。"
        )

    llm = ChatOpenAI(
        model=args.judge_model,
        base_url=api_base,
        api_key=api_key,
        temperature=0.0,
        timeout=args.timeout,
    )
    return LangchainLLMWrapper(llm)


def build_judge_embeddings() -> LocalBGEEmbeddings:
    settings = Settings.from_env()
    embedder = BGEM3Embedder(
        model_path=settings.kb.embedding_model_path,
        dim=settings.kb.embedding_dim,
    )
    return LocalBGEEmbeddings(embedder)


def summarize_scores(score_rows: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = [key for key in score_rows[0].keys() if key != "user_input"]
    summary: dict[str, float] = {}
    for name in metric_names:
        values = [
            row[name]
            for row in score_rows
            if isinstance(row.get(name), (int, float)) and not _is_nan(row[name])
        ]
        if values:
            summary[name] = round(mean(values), 4)
    return summary


def mean_or_none(values: list[float]) -> float | None:
    numeric = [value for value in values if isinstance(value, (int, float))]
    if not numeric:
        return None
    return round(mean(numeric), 4)


def overlap_metrics(expected: list[str], actual: list[str]) -> dict[str, float | list[str] | bool]:
    expected_set = {item for item in expected if item}
    actual_set = {item for item in actual if item}
    overlap = sorted(expected_set & actual_set)
    precision = len(overlap) / len(actual_set) if actual_set else 0.0
    recall = len(overlap) / len(expected_set) if expected_set else 0.0
    hit = bool(overlap) if expected_set else False
    return {
        "expected": sorted(expected_set),
        "actual": sorted(actual_set),
        "matched": overlap,
        "hit": hit,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def overlap_metrics_with_accepts(
    expected: list[str],
    accepted: list[str],
    actual: list[str],
) -> dict[str, float | list[str] | bool]:
    primary_expected = {item for item in expected if item}
    accepted_set = {item for item in accepted if item}
    actual_set = {item for item in actual if item}
    eligible = primary_expected | accepted_set
    overlap = sorted(eligible & actual_set)
    precision = len(overlap) / len(actual_set) if actual_set else 0.0
    recall = len(overlap) / len(eligible) if eligible else 0.0
    hit = bool(overlap) if eligible else False
    return {
        "expected": sorted(primary_expected),
        "accepted": sorted(accepted_set),
        "eligible": sorted(eligible),
        "actual": sorted(actual_set),
        "matched": overlap,
        "hit": hit,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def evaluate_evidence_coverage(
    reference: str,
    contexts: list[str],
    embeddings: BaseRagasEmbeddings | None = None,
) -> dict[str, Any]:
    facts = extract_answer_facts(reference)
    context_text = "\n".join(contexts)
    covered: list[str] = []
    missing: list[str] = []
    for fact in facts:
        if fact_present_in_context(fact, context_text, embeddings=embeddings):
            covered.append(fact)
        else:
            missing.append(fact)
    coverage = len(covered) / len(facts) if facts else None
    return {
        "answer_fact_count": len(facts),
        "covered_fact_count": len(covered),
        "evidence_coverage": round(coverage, 4) if coverage is not None else None,
        "covered_facts": covered,
        "missing_facts": missing,
    }


def get_evidence_quality(item: dict[str, Any]) -> dict[str, Any]:
    return ((item.get("api_response") or {}).get("debug") or {}).get("evidence_quality") or {}


def get_final_answer_mode(item: dict[str, Any]) -> str:
    return str(get_evidence_quality(item).get("final_answer_mode") or "").strip()


def get_support_pack_count(item: dict[str, Any]) -> int:
    return normalize_int(get_evidence_quality(item).get("support_pack_count"))


def get_candidate_support_pack_count(item: dict[str, Any]) -> int:
    evidence = get_evidence_quality(item)
    if "candidate_support_pack_count" in evidence:
        return normalize_int(evidence.get("candidate_support_pack_count"))
    return get_support_pack_count(item)


def get_should_refuse_final(item: dict[str, Any]) -> bool:
    return bool(get_evidence_quality(item).get("should_refuse_final"))


def get_refusal_reason(item: dict[str, Any]) -> str:
    return str(get_evidence_quality(item).get("refusal_reason") or "").strip()


def get_answer_preview(item: dict[str, Any], limit: int = 120) -> str:
    response = re.sub(r"\s+", " ", str(item.get("response") or "")).strip()
    return response[:limit]


def is_effective_refusal(item: dict[str, Any]) -> bool:
    final_answer_mode = get_final_answer_mode(item)
    if final_answer_mode == "limited_partial_compare":
        return False
    if final_answer_mode == "refuse":
        return True
    if get_should_refuse_final(item):
        return True
    return _is_abstention_response(str(item.get("response") or ""))


def get_effective_final_answer_mode(item: dict[str, Any]) -> str:
    if is_effective_refusal(item):
        return "refuse"
    final_answer_mode = get_final_answer_mode(item)
    return final_answer_mode or "unknown"


def is_substantive_answer(item: dict[str, Any]) -> bool:
    return not is_effective_refusal(item)


def has_missing_branch_disclosure(response: str) -> bool:
    lowered = (response or "").lower()
    return any(
        token in lowered
        for token in (
            "未覆盖",
            "未召回",
            "缺失",
            "证据不足",
            "未检索到",
            "当前缺少",
            "无法完成完整对比",
            "无法可靠作答",
            "未提供",
            "未找到",
        )
    )


def has_partial_compare_conclusion(response: str) -> bool:
    text = re.sub(r"\s+", " ", response or "").strip()
    return "有限比较结论" in text and "可以确认" in text


def compute_branch_metrics(
    meta: dict[str, Any],
    actual_doc_ids: list[str],
    actual_source_files: list[str],
    item: dict[str, Any],
) -> dict[str, Any]:
    evidence = get_evidence_quality(item)
    branches = meta.get("comparison_branches") or []
    branch_count = len(branches)
    if branch_count == 0:
        return {
            "branch_count": 0,
            "covered_branch_count": 0,
            "branch_doc_coverage": None,
            "missing_branch_count": 0,
            "citation_doc_coverage": None,
            "partial_compare": False,
            "unsupported_branch_claim_count": 0,
            "missing_branch_disclosed_count": 0,
            "reasonable_refusal": False,
            "false_refusal": False,
        }

    covered_branch_count = 0
    missing_branch_count = 0
    missing_branch_disclosed_count = 0
    response = str(item.get("response") or "")
    did_refuse = is_effective_refusal(item)
    disclosure = has_missing_branch_disclosure(response)
    actual_doc_set = {doc for doc in actual_doc_ids if doc}
    actual_source_set = {src for src in actual_source_files if src}
    matched_doc_ids: set[str] = set()
    matched_source_files: set[str] = set()

    for branch in branches:
        branch_docs = normalize_list(branch.get("expected_doc_ids"))
        branch_accepted_docs = normalize_list(branch.get("accepted_doc_ids"))
        branch_sources = normalize_list(branch.get("expected_source_files"))
        branch_accepted_sources = normalize_list(branch.get("accepted_source_files"))
        eligible_docs = set(branch_docs) | set(branch_accepted_docs)
        eligible_sources = set(branch_sources) | set(branch_accepted_sources)
        covered = bool((eligible_docs & actual_doc_set) or (eligible_sources & actual_source_set))
        if covered:
            covered_branch_count += 1
            matched_doc_ids.update(eligible_docs & actual_doc_set)
            matched_source_files.update(eligible_sources & actual_source_set)
        else:
            missing_branch_count += 1
            if disclosure:
                missing_branch_disclosed_count += 1

    branch_doc_coverage = round(covered_branch_count / branch_count, 4) if branch_count else None
    total_expected_docs = {
        doc_id
        for branch in branches
        for doc_id in normalize_list(branch.get("expected_doc_ids")) + normalize_list(branch.get("accepted_doc_ids"))
        if doc_id
    }
    total_expected_sources = {
        src
        for branch in branches
        for src in normalize_list(branch.get("expected_source_files")) + normalize_list(branch.get("accepted_source_files"))
        if src
    }
    if total_expected_docs:
        citation_doc_coverage = round(len(matched_doc_ids) / len(total_expected_docs), 4)
    elif total_expected_sources:
        citation_doc_coverage = round(len(matched_source_files) / len(total_expected_sources), 4)
    else:
        citation_doc_coverage = None

    allow_partial_threshold = normalize_int(meta.get("allow_partial_if_doc_coverage"))
    partial_allowed = 0 < allow_partial_threshold <= covered_branch_count < normalize_int(meta.get("expected_min_doc_coverage"))
    final_answer_mode = get_effective_final_answer_mode(item)
    partial_compare = final_answer_mode in {"partial", "limited_partial", "limited_partial_compare"} or (
        not did_refuse and partial_allowed and disclosure
    )
    unsupported_branch_claim_count = normalize_int(
        evidence.get("unsupported_branch_claim_count"),
        missing_branch_count if (missing_branch_count > 0 and not did_refuse and not disclosure) else 0,
    )
    missing_branch_disclosed_count = normalize_int(
        evidence.get("missing_branch_disclosed"),
        missing_branch_disclosed_count,
    )
    if isinstance(evidence.get("missing_branch_disclosed"), bool):
        missing_branch_disclosed_count = missing_branch_count if evidence.get("missing_branch_disclosed") else 0

    return {
        "branch_count": branch_count,
        "covered_branch_count": covered_branch_count,
        "branch_doc_coverage": branch_doc_coverage,
        "missing_branch_count": missing_branch_count,
        "citation_doc_coverage": citation_doc_coverage,
        "partial_compare": partial_compare,
        "unsupported_branch_claim_count": unsupported_branch_claim_count,
        "missing_branch_disclosed_count": missing_branch_disclosed_count,
        "covered_branch_but_refuse": bool(evidence.get("covered_branch_but_refuse")),
        "branch_support_pack_counts": evidence.get("branch_support_pack_counts") or {},
        "branch_citable_quote_counts": evidence.get("branch_citable_quote_counts") or {},
        "branch_labels": [str(branch.get("label") or "").strip() for branch in branches],
        "missing_branch_labels": [
            str(branch.get("label") or "").strip()
            for branch in branches
            if not (
                (set(normalize_list(branch.get("expected_doc_ids"))) | set(normalize_list(branch.get("accepted_doc_ids"))))
                & actual_doc_set
                or (set(normalize_list(branch.get("expected_source_files"))) | set(normalize_list(branch.get("accepted_source_files"))))
                & actual_source_set
            )
        ],
        "reasonable_refusal": False,
        "false_refusal": False,
    }


def classify_refusal(
    item: dict[str, Any],
    meta: dict[str, Any],
    doc_metrics: dict[str, Any],
    branch_metrics: dict[str, Any],
    failure_type: str,
    citation_count: int,
) -> dict[str, Any]:
    did_refuse = is_effective_refusal(item)
    support_pack_count = get_support_pack_count(item)
    candidate_support_pack_count = get_candidate_support_pack_count(item)
    refusal_reason = get_refusal_reason(item)
    evidence = get_evidence_quality(item)
    route = str((item.get("api_response") or {}).get("route") or "").strip()
    allow_partial_threshold = normalize_int(meta.get("allow_partial_if_doc_coverage"))
    covered_branch_count = normalize_int(branch_metrics.get("covered_branch_count"))
    comparison_partial_available = (
        route == "comparison"
        and allow_partial_threshold > 0
        and covered_branch_count >= allow_partial_threshold
    )
    no_citable_evidence = citation_count == 0
    retrieval_guardrail_hit = bool(
        evidence.get("empty_support_pack_guardrail_triggered")
        or evidence.get("zero_citation_guardrail_triggered")
        or evidence.get("fallback_guardrail_failed")
    )
    no_context_guardrail_hit = normalize_int(((item.get("api_response") or {}).get("debug") or {}).get("final_context_count")) == 0

    reasonable_refusal = did_refuse and (
        support_pack_count == 0
        or candidate_support_pack_count == 0
        or no_citable_evidence
        or failure_type in {"retrieval_failure", "evidence_failure"}
        or retrieval_guardrail_hit
        or no_context_guardrail_hit
        or (
            route == "comparison"
            and normalize_int(branch_metrics.get("branch_count")) > 0
            and covered_branch_count == 0
        )
        or refusal_reason in {
            "empty_support_refuse",
            "missing_all_branches_refuse",
            "post_generation_zero_citation_refuse",
            "fallback_without_citable_support",
            "weak_evidence",
        }
    )

    false_refusal = did_refuse and not reasonable_refusal and (
        (bool(doc_metrics.get("hit")) and support_pack_count > 0)
        or comparison_partial_available
        or candidate_support_pack_count > 0
        or refusal_reason == "covered_branch_but_refuse"
    )
    unclassified_refusal = did_refuse and not reasonable_refusal and not false_refusal

    return {
        "did_refuse": did_refuse,
        "support_pack_count": support_pack_count,
        "candidate_support_pack_count": candidate_support_pack_count,
        "refusal_reason": refusal_reason,
        "reasonable_refusal": reasonable_refusal,
        "false_refusal": false_refusal,
        "unclassified_refusal": unclassified_refusal,
        "covered_branch_but_refuse": refusal_reason == "covered_branch_but_refuse",
        "refusal_no_citation": did_refuse and citation_count == 0,
        "no_context_guardrail_triggered": no_context_guardrail_hit,
    }


def extract_answer_facts(reference: str) -> list[str]:
    cleaned = re.sub(r"^回答应(?:指出|提到|说明|概括)?", "", reference or "").strip(" ：:，,。")
    cleaned = cleaned.replace("不要扩展到文库未提及的家族", "")
    parts = [
        item.strip(" ：:，,。")
        for item in re.split(r"[；;。]\s*", cleaned)
        if item.strip(" ：:，,。")
    ]
    facts: list[str] = []
    for part in parts:
        if len(part) <= 80:
            facts.append(part)
            continue
        subparts = [item.strip(" ：:，,。") for item in re.split(r"[，,]\s*", part) if item.strip(" ：:，,。")]
        facts.extend(item for item in subparts if len(item) >= 4)
    return dedupe_preserve_order(facts)


def fact_present_in_context(
    fact: str,
    context_text: str,
    embeddings: BaseRagasEmbeddings | None = None,
) -> bool:
    fact_norm = normalize_fact_text(fact)
    context_norm = normalize_fact_text(context_text)
    if not fact_norm or not context_norm:
        return False

    if fact_norm in context_norm:
        return True

    fact_alias_norm = normalize_aliases(fact_norm)
    context_alias_norm = normalize_aliases(context_norm)
    if fact_alias_norm and fact_alias_norm in context_alias_norm:
        return True

    fact_numbers = extract_numbers_and_units(fact_norm)
    context_numbers = extract_numbers_and_units(context_norm)
    number_overlap_ratio = compute_term_overlap_ratio(fact_numbers, context_numbers)
    has_number_anchor = bool(fact_numbers) and number_overlap_ratio >= 0.5

    fact_entities = extract_entities(fact_norm)
    context_entities = extract_entities(context_norm)
    entity_overlap_ratio = compute_term_overlap_ratio(fact_entities, context_entities)
    has_entity_anchor = bool(fact_entities) and entity_overlap_ratio >= 0.5

    fact_terms = set(extract_fact_terms(fact_alias_norm))
    context_terms = set(extract_fact_terms(context_alias_norm))
    term_overlap_ratio = compute_term_overlap_ratio(fact_terms, context_terms)

    if has_number_anchor and (term_overlap_ratio >= 0.15 or has_entity_anchor):
        return True
    if has_entity_anchor and term_overlap_ratio >= 0.2:
        return True
    if term_overlap_ratio >= 0.55:
        return True

    return semantic_fact_present_in_context(
        fact=fact_norm,
        context_text=context_norm,
        embeddings=embeddings,
        has_number_anchor=has_number_anchor,
        has_entity_anchor=has_entity_anchor,
        term_overlap_ratio=term_overlap_ratio,
    )


def normalize_fact_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace("’", "'").replace("′", "'")
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = text.replace("μ", "u").replace("µ", "u")
    text = text.replace("α", "alpha").replace("β", "beta").replace("γ", "gamma")
    text = text.replace("≤", "<=").replace("≥", ">=")
    text = re.sub(r"[（\(]\s*([a-z0-9][a-z0-9+_.:,/-]{1,})\s*[）\)]", r" \1 ", text)
    text = re.sub(r"[，。；：、]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_fact_terms(text: str) -> list[str]:
    text = normalize_aliases(text)

    ascii_terms = re.findall(r"[a-z][a-z0-9_/-]*", text)
    cjk_terms = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    cjk_bigrams = ["".join(cjk_chars[i:i + 2]) for i in range(max(len(cjk_chars) - 1, 0))]
    number_terms = sorted(extract_numbers_and_units(text))

    merged = ascii_terms + cjk_terms + cjk_bigrams + number_terms
    filtered = [
        term for term in merged
        if len(term) >= 2 and term not in _FACT_STOPWORDS
    ]
    return dedupe_preserve_order(filtered)


_TERM_CANONICAL_MAP = {
    # 中文 <-> 英文常见科研术语
    "比生长速率": "specific_growth_rate",
    "生长速率": "growth_rate",
    "specific growth rate": "specific_growth_rate",
    "growth rate": "growth_rate",

    "分泌速率": "secretion_rate",
    "secretory rate": "secretion_rate",
    "secretion rate": "secretion_rate",

    "产率": "yield",
    "得率": "yield",
    "yield": "yield",

    "生产强度": "productivity",
    "生产率": "productivity",
    "productivity": "productivity",
    "titer": "titer",

    "木糖激酶": "xylosyl_kinase",
    "xylosyl kinase": "xylosyl_kinase",

    "岩藻糖基转移酶": "fucosyltransferase",
    "fucosyltransferase": "fucosyltransferase",
    "alpha1,2-fucosyltransferase": "fucosyltransferase",
    "alpha-1,2-fucosyltransferase": "fucosyltransferase",

    "唾液酸转移酶": "sialyltransferase",
    "sialyltransferase": "sialyltransferase",

    "唾液酸": "neu5ac",
    "n-acetylneuraminic acid": "neu5ac",
    "neu5ac": "neu5ac",

    "高溶解度": "high_solubility",
    "solubility": "solubility",

    "高度保守": "highly_conserved",
    "保守": "conserved",
    "highly conserved": "highly_conserved",
    "conserved": "conserved",

    "糖胺聚糖": "glycosaminoglycan",
    "glycosaminoglycan": "glycosaminoglycan",

    "膜转运": "membrane_transport",
    "膜转运家族": "transporter_family",
    "膜转运蛋白": "membrane_transporter",
    "转运家族": "transporter_family",
    "transporter family": "transporter_family",
    "membrane transporter": "membrane_transporter",
    "major facilitator superfamily": "mfs",
    "mfs": "mfs",
    "atp-binding cassette": "abc",
    "atp binding cassette": "abc",
    "abc": "abc",
    "resistance-nodulation-division": "rnd",
    "resistance nodulation division": "rnd",
    "rnd": "rnd",
    "small multidrug resistance": "smr",
    "smr": "smr",
    "multidrug and toxic compound extrusion": "mate",
    "mate": "mate",
    "芳香化合物": "aromatic_compound",
    "aromatic compounds": "aromatic_compound",

    "证据不足": "insufficient_evidence",
    "insufficient evidence": "insufficient_evidence",
}


def canonicalize_text(text: str) -> str:
    return normalize_aliases(text)


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def normalize_aliases(text: str) -> str:
    text = normalize_fact_text(text)
    for src, tgt in sorted(_TERM_CANONICAL_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        src_norm = normalize_fact_text(src)
        if not src_norm:
            continue
        if contains_cjk(src_norm):
            text = text.replace(src_norm, f" {tgt} ")
        else:
            text = re.sub(rf"(?<![a-z0-9_]){re.escape(src_norm)}(?![a-z0-9_])", f" {tgt} ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_numbers_and_units(text: str) -> set[str]:
    text = normalize_fact_text(text)
    unit_pattern = r"%|g/l/h|g/l|mg/l|mg|mmol|mm|um|umol|nm|fold|h|hr|hours?|times?|x"
    anchors: set[str] = set()
    for start, end, unit in re.findall(
        rf"\b(\d+(?:\.\d+)?)\s*(?:-|~|to)\s*(\d+(?:\.\d+)?)\s*({unit_pattern})\b",
        text,
    ):
        unit_norm = _normalize_unit(unit)
        start_norm = _normalize_number(start)
        end_norm = _normalize_number(end)
        anchors.add(f"{start_norm}{unit_norm}")
        anchors.add(f"{end_norm}{unit_norm}")
        anchors.add(f"{start_norm}-{end_norm}{unit_norm}")
    for value, unit in re.findall(rf"\b(\d+(?:\.\d+)?)\s*({unit_pattern})?\b", text):
        anchors.add(f"{_normalize_number(value)}{_normalize_unit(unit)}")
    return {anchor for anchor in anchors if anchor}


def extract_entities(text: str) -> set[str]:
    normalized = normalize_aliases(text)
    entities = set(
        re.findall(
            r"\b[a-z]{1,8}\d{1,5}[a-z0-9-]*\b|"
            r"\b[a-z0-9]+(?:_[a-z0-9]+)+\b|"
            r"\b[a-z]{2,18}ase\b",
            normalized,
        )
    )
    known_entity_terms = {
        "abc",
        "mfs",
        "rnd",
        "smr",
        "mate",
        "neu5ac",
        "hsa",
        "bge",
    }
    entities.update(
        token for token in re.findall(r"\b[a-z]{2,8}\b", normalized)
        if token in known_entity_terms
    )
    cjk_entities = re.findall(
        r"[\u4e00-\u9fff]{2,}(?:酶|酸|菌|家族|转运体|转运蛋白|通道|泵|系统)",
        normalize_fact_text(text),
    )
    entities.update(normalize_aliases(entity) for entity in cjk_entities)
    return {
        entity for entity in entities
        if len(entity) >= 2 and entity not in _FACT_STOPWORDS and not entity.isdigit()
    }


def extract_named_entities_like_terms(text: str) -> set[str]:
    return extract_entities(text)


def term_overlap_score(fact_terms: set[str], context_terms: set[str]) -> float:
    return compute_term_overlap_ratio(fact_terms, context_terms)


def compute_term_overlap_ratio(fact_terms: set[str], context_terms: set[str]) -> float:
    if not fact_terms:
        return 0.0
    overlap = fact_terms & context_terms
    return len(overlap) / len(fact_terms)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def semantic_fact_present_in_context(
    fact: str,
    context_text: str,
    embeddings: BaseRagasEmbeddings | None = None,
    has_number_anchor: bool = False,
    has_entity_anchor: bool = False,
    term_overlap_ratio: float = 0.0,
) -> bool:
    if embeddings is None:
        return False
    fact_for_embedding = normalize_aliases(fact)
    context_for_embedding = normalize_aliases(context_text)
    if not fact_for_embedding or not context_for_embedding:
        return False
    segments = [
        segment.strip()
        for segment in re.split(r"[\n。；;.!?]\s*", context_for_embedding)
        if len(segment.strip()) >= 12
    ]
    segments = dedupe_preserve_order(segments)[:24] or [context_for_embedding[:4000]]
    try:
        fact_vector = embeddings.embed_query(fact_for_embedding)
        context_vectors = embeddings.embed_documents([segment[:800] for segment in segments])
    except Exception as exc:
        print(f"[warn] evidence semantic fallback failed: {exc}", file=sys.stderr)
        return False
    similarity = max(cosine_similarity(fact_vector, vector) for vector in context_vectors)
    if (has_number_anchor or has_entity_anchor) and similarity >= 0.58:
        return True
    if term_overlap_ratio >= 0.25 and similarity >= 0.62:
        return True
    return similarity >= 0.72


def _normalize_number(value: str) -> str:
    try:
        parsed = float(value)
    except ValueError:
        return value
    if parsed.is_integer():
        return str(int(parsed))
    return str(parsed).rstrip("0").rstrip(".")


def _normalize_unit(unit: str) -> str:
    unit = normalize_fact_text(unit)
    if unit in {"hr", "hour", "hours"}:
        return "h"
    if unit in {"time", "times", "x"}:
        return "fold"
    return unit


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


_FACT_STOPWORDS = {
    "回答",
    "应指",
    "指出",
    "说明",
    "概括",
    "包括",
    "主要",
    "以及",
    "利用",
    "研究",
    "文库",
    "证据",
    "系统",
    "相关",
    "通过",
    "进行",
    "实现",
    "其中",
    "最终",
    "表明",
    "提示",
    "具有",
    "作用",
    "功能",
    "机制",
    "策略",
    "问题",
    "工作",
    "内容",
    "比较",
    "强调",
    "需要",
    "可以",
    "用于",
    "关于",
    "当前",
    "理想回答",
    "应明确说明",

    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "with",
    "for",
    "from",
    "by",
    "on",
    "at",
    "an",
    "a",
    "this",
    "that",
    "these",
    "those",
    "study",
    "results",
    "discussion",
    "abstract",
    "introduction",
    "conclusion",
    "full",
    "text",
    "materials",
    "methods",
}


def evaluate_retrieval(
    records: list[dict[str, Any]],
    embeddings: BaseRagasEmbeddings | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_item: list[dict[str, Any]] = []
    citation_count_values: list[float] = []
    min_citation_pass_all_values: list[float] = []
    min_citation_pass_substantive_values: list[float] = []
    strict_route_match_values: list[float] = []
    accepted_route_match_values: list[float] = []
    doc_hit_values: list[float] = []
    doc_precision_values: list[float] = []
    doc_recall_values: list[float] = []
    source_hit_values: list[float] = []
    source_precision_values: list[float] = []
    source_recall_values: list[float] = []
    section_hit_values: list[float] = []
    section_precision_values: list[float] = []
    section_recall_values: list[float] = []
    evidence_coverage_values: list[float] = []
    branch_count_values: list[float] = []
    covered_branch_count_values: list[float] = []
    branch_doc_coverage_values: list[float] = []
    citation_doc_coverage_values: list[float] = []
    partial_compare_values: list[float] = []
    unsupported_branch_claim_values: list[float] = []
    missing_branch_disclosed_values: list[float] = []
    reasonable_refusal_values: list[float] = []
    false_refusal_values: list[float] = []
    unclassified_refusal_values: list[float] = []
    zero_citation_substantive_values: list[float] = []
    refusal_no_citation_values: list[float] = []
    limited_partial_values: list[float] = []

    for item in records:
        citations = item["api_response"].get("citations") or []
        actual_doc_ids = [str(citation.get("doc_id") or "").strip() for citation in citations if citation.get("doc_id")]
        actual_source_files = [
            str(citation.get("source_file") or "").strip()
            for citation in citations
            if citation.get("source_file")
        ]
        actual_sections = [str(citation.get("section") or "").strip() for citation in citations if citation.get("section")]

        meta = item["dataset_meta"]
        citation_count = len(citations)
        expected_min_citations = meta["expected_min_citations"]
        route = str(item["api_response"].get("route") or "").strip()
        expected_route = meta["expected_route"]
        accepted_routes = meta.get("accepted_routes") or []

        doc_metrics = overlap_metrics_with_accepts(
            meta["expected_doc_ids"],
            meta.get("accepted_doc_ids") or [],
            actual_doc_ids,
        )
        source_metrics = overlap_metrics_with_accepts(
            meta["expected_source_files"],
            meta.get("accepted_source_files") or [],
            actual_source_files,
        )
        section_metrics = overlap_metrics(meta["expected_sections"], actual_sections)
        evidence_metrics = evaluate_evidence_coverage(
            reference=item.get("reference") or "",
            contexts=item.get("retrieved_contexts") or [],
            embeddings=embeddings,
        )
        min_citation_pass = citation_count >= expected_min_citations if expected_min_citations > 0 else True
        strict_route_match = route == expected_route if expected_route else None
        if accepted_routes:
            accepted_route_match = route in accepted_routes
        else:
            accepted_route_match = strict_route_match
        failure_type = classify_retrieval_failure(doc_metrics, evidence_metrics)
        branch_metrics = compute_branch_metrics(meta, actual_doc_ids, actual_source_files, item)
        zero_citation_substantive = citation_count == 0 and is_substantive_answer(item)
        final_answer_mode = get_effective_final_answer_mode(item)
        refusal_metrics = classify_refusal(
            item=item,
            meta=meta,
            doc_metrics=doc_metrics,
            branch_metrics=branch_metrics,
            failure_type=failure_type,
            citation_count=citation_count,
        )

        retrieval_eval = {
            "citation_count": citation_count,
            "expected_min_citations": expected_min_citations,
            "min_citation_pass": min_citation_pass,
            "min_citation_pass_substantive": (
                min_citation_pass if is_substantive_answer(item) else None
            ),
            "route": route,
            "expected_route": expected_route,
            "accepted_routes": accepted_routes,
            "strict_route_match": strict_route_match,
            "route_match": strict_route_match,
            "accepted_route_match": accepted_route_match,
            "doc_id_metrics": doc_metrics,
            "source_file_metrics": source_metrics,
            "section_metrics": section_metrics,
            "evidence_metrics": evidence_metrics,
            "failure_type": failure_type,
            "final_answer_mode": final_answer_mode,
            "zero_citation_substantive_answer": zero_citation_substantive,
            **branch_metrics,
            **refusal_metrics,
        }
        item["retrieval_eval"] = retrieval_eval
        per_item.append({"id": item["id"], **retrieval_eval})

        citation_count_values.append(float(citation_count))
        min_citation_pass_all_values.append(1.0 if min_citation_pass else 0.0)
        if is_substantive_answer(item):
            min_citation_pass_substantive_values.append(1.0 if min_citation_pass else 0.0)
        if strict_route_match is not None:
            strict_route_match_values.append(1.0 if strict_route_match else 0.0)
        if accepted_route_match is not None:
            accepted_route_match_values.append(1.0 if accepted_route_match else 0.0)
        if doc_metrics["eligible"]:
            doc_hit_values.append(1.0 if doc_metrics["hit"] else 0.0)
            doc_precision_values.append(float(doc_metrics["precision"]))
            doc_recall_values.append(float(doc_metrics["recall"]))
        if source_metrics["eligible"]:
            source_hit_values.append(1.0 if source_metrics["hit"] else 0.0)
            source_precision_values.append(float(source_metrics["precision"]))
            source_recall_values.append(float(source_metrics["recall"]))
        if meta["expected_sections"]:
            section_hit_values.append(1.0 if section_metrics["hit"] else 0.0)
            section_precision_values.append(float(section_metrics["precision"]))
            section_recall_values.append(float(section_metrics["recall"]))
        if isinstance(evidence_metrics.get("evidence_coverage"), (int, float)):
            evidence_coverage_values.append(float(evidence_metrics["evidence_coverage"]))
        if isinstance(branch_metrics.get("branch_count"), int) and branch_metrics["branch_count"] > 0:
            branch_count_values.append(float(branch_metrics["branch_count"]))
            covered_branch_count_values.append(float(branch_metrics["covered_branch_count"]))
            if isinstance(branch_metrics.get("branch_doc_coverage"), (int, float)):
                branch_doc_coverage_values.append(float(branch_metrics["branch_doc_coverage"]))
            if isinstance(branch_metrics.get("citation_doc_coverage"), (int, float)):
                citation_doc_coverage_values.append(float(branch_metrics["citation_doc_coverage"]))
            partial_compare_values.append(1.0 if branch_metrics.get("partial_compare") else 0.0)
            unsupported_branch_claim_values.append(float(branch_metrics.get("unsupported_branch_claim_count") or 0.0))
            missing_branch_disclosed_values.append(float(branch_metrics.get("missing_branch_disclosed_count") or 0.0))
        reasonable_refusal_values.append(1.0 if refusal_metrics.get("reasonable_refusal") else 0.0)
        false_refusal_values.append(1.0 if refusal_metrics.get("false_refusal") else 0.0)
        unclassified_refusal_values.append(1.0 if refusal_metrics.get("unclassified_refusal") else 0.0)
        zero_citation_substantive_values.append(1.0 if zero_citation_substantive else 0.0)
        refusal_no_citation_values.append(1.0 if refusal_metrics.get("refusal_no_citation") else 0.0)
        limited_partial_values.append(1.0 if final_answer_mode in {"limited_partial", "limited_partial_compare"} else 0.0)

    summary = {
        "citation_count_avg": mean_or_none(citation_count_values),
        "min_citation_pass_rate": mean_or_none(min_citation_pass_all_values),
        "min_citation_pass_rate_all": mean_or_none(min_citation_pass_all_values),
        "min_citation_pass_rate_substantive_only": mean_or_none(min_citation_pass_substantive_values),
        "strict_route_match_rate": mean_or_none(strict_route_match_values),
        "route_match_rate": mean_or_none(strict_route_match_values),
        "accepted_route_match_rate": mean_or_none(accepted_route_match_values),
        "doc_id_hit_rate": mean_or_none(doc_hit_values),
        "doc_id_precision_avg": mean_or_none(doc_precision_values),
        "doc_id_recall_avg": mean_or_none(doc_recall_values),
        "source_file_hit_rate": mean_or_none(source_hit_values),
        "source_file_precision_avg": mean_or_none(source_precision_values),
        "source_file_recall_avg": mean_or_none(source_recall_values),
        "section_hit_rate": mean_or_none(section_hit_values),
        "section_precision_avg": mean_or_none(section_precision_values),
        "section_recall_avg": mean_or_none(section_recall_values),
        "evidence_coverage_avg": mean_or_none(evidence_coverage_values),
        "evidence_coverage_zero_rate": mean_or_none([1.0 if value == 0.0 else 0.0 for value in evidence_coverage_values]),
        "branch_count_avg": mean_or_none(branch_count_values),
        "covered_branch_count_avg": mean_or_none(covered_branch_count_values),
        "branch_doc_coverage_avg": mean_or_none(branch_doc_coverage_values),
        "citation_doc_coverage_avg": mean_or_none(citation_doc_coverage_values),
        "partial_compare_rate": mean_or_none(partial_compare_values),
        "unsupported_branch_claim_count_avg": mean_or_none(unsupported_branch_claim_values),
        "missing_branch_disclosed_count_avg": mean_or_none(missing_branch_disclosed_values),
        "reasonable_refusal_rate": mean_or_none(reasonable_refusal_values),
        "false_refusal_rate": mean_or_none(false_refusal_values),
        "unclassified_refusal_rate": mean_or_none(unclassified_refusal_values),
        "zero_citation_substantive_answer_rate": mean_or_none(zero_citation_substantive_values),
        "refusal_no_citation_rate": mean_or_none(refusal_no_citation_values),
        "limited_partial_rate": mean_or_none(limited_partial_values),
    }
    return per_item, {key: value for key, value in summary.items() if value is not None}


def classify_retrieval_failure(doc_metrics: dict[str, Any], evidence_metrics: dict[str, Any]) -> str:
    if doc_metrics.get("expected") and not doc_metrics.get("hit"):
        return "retrieval_failure"
    evidence_coverage = evidence_metrics.get("evidence_coverage")
    if isinstance(evidence_coverage, (int, float)) and evidence_coverage == 0:
        return "evidence_failure"
    return "none"


def summarize_generation_from_items(items: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = sorted(
        {
            key
            for item in items
            for key, value in item.items()
            if key != "id" and isinstance(value, (int, float))
        }
    )
    summary: dict[str, float] = {}
    for name in metric_names:
        values = [float(item[name]) for item in items if isinstance(item.get(name), (int, float))]
        if values:
            summary[name] = round(mean(values), 4)
    return summary


def build_slice_summary(
    records: list[dict[str, Any]],
    embeddings: BaseRagasEmbeddings | None = None,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        "tags": {},
        "scenario": {},
        "ability": {},
        "difficulty": {},
        "risk_level": {},
        "expected_behavior": {},
    }

    for item in records:
        meta = item["dataset_meta"]
        for tag in meta.get("tags") or []:
            grouped["tags"].setdefault(tag, []).append(item)
        for field in ("scenario", "ability", "difficulty", "risk_level", "expected_behavior"):
            value = str(meta.get(field) or "").strip()
            if value:
                grouped[field].setdefault(value, []).append(item)

    slices: dict[str, Any] = {}
    for dimension, buckets in grouped.items():
        dimension_summary: dict[str, Any] = {}
        for bucket, items in sorted(buckets.items()):
            retrieval_items = [{"id": item["id"], **item["retrieval_eval"]} for item in items if item.get("retrieval_eval")]
            generation_items = [{"id": item["id"], **item["generation_eval"]} for item in items if item.get("generation_eval")]
            _, retrieval_summary = evaluate_retrieval(items, embeddings=embeddings)
            dimension_summary[bucket] = {
                "sample_count": len(items),
                "retrieval": retrieval_summary,
                "generation": summarize_generation_from_items(generation_items) if generation_items else {},
                "sample_ids": [item["id"] for item in items],
            }
        if dimension_summary:
            slices[dimension] = dimension_summary
    return slices


def metric_value(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def gate_status(actual: float | None, threshold: float, comparator: str = ">=") -> str:
    if actual is None:
        return "not_applicable"
    if comparator == ">=":
        return "pass" if actual >= threshold else "fail"
    return "pass" if actual <= threshold else "fail"


def build_enterprise_gates(report: dict[str, Any]) -> dict[str, Any]:
    retrieval_summary = report.get("retrieval", {}).get("summary", {})
    generation_summary = report.get("generation", {}).get("summary", {})
    slices = report.get("enterprise_slices", {})
    high_risk = slices.get("risk_level", {}).get("high", {})
    high_risk_retrieval = high_risk.get("retrieval", {})
    high_risk_generation = high_risk.get("generation", {})

    gates = {
        "sample_count_min_20": {
            "actual": report.get("sample_count"),
            "threshold": 20,
            "status": "pass" if int(report.get("sample_count") or 0) >= 20 else "fail",
        },
        "route_match_rate": {
            "actual": metric_value(retrieval_summary, "route_match_rate"),
            "threshold": 0.7,
            "status": gate_status(metric_value(retrieval_summary, "route_match_rate"), 0.7),
        },
        "min_citation_pass_rate": {
            "actual": metric_value(retrieval_summary, "min_citation_pass_rate"),
            "threshold": 0.8,
            "status": gate_status(metric_value(retrieval_summary, "min_citation_pass_rate"), 0.8),
        },
        "doc_id_hit_rate": {
            "actual": metric_value(retrieval_summary, "doc_id_hit_rate"),
            "threshold": 0.7,
            "status": gate_status(metric_value(retrieval_summary, "doc_id_hit_rate"), 0.7),
        },
        "faithfulness": {
            "actual": metric_value(generation_summary, "faithfulness"),
            "threshold": 0.75,
            "status": gate_status(metric_value(generation_summary, "faithfulness"), 0.75),
        },
        "answer_relevancy": {
            "actual": metric_value(generation_summary, "answer_relevancy"),
            "threshold": 0.75,
            "status": gate_status(metric_value(generation_summary, "answer_relevancy"), 0.75),
        },
        "high_risk_doc_hit_rate": {
            "actual": metric_value(high_risk_retrieval, "doc_id_hit_rate"),
            "threshold": 0.8,
            "status": gate_status(metric_value(high_risk_retrieval, "doc_id_hit_rate"), 0.8),
        },
        "high_risk_faithfulness": {
            "actual": metric_value(high_risk_generation, "faithfulness"),
            "threshold": 0.8,
            "status": gate_status(metric_value(high_risk_generation, "faithfulness"), 0.8),
        },
    }
    overall = "pass"
    for item in gates.values():
        if item["status"] == "fail":
            overall = "fail"
            break
    return {
        "overall_status": overall,
        "gates": gates,
    }


def build_failure_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    zero_citation_ids: list[str] = []
    zero_citation_substantive_answer_ids: list[str] = []
    refusal_no_citation_ids: list[str] = []
    strict_route_miss_ids: list[str] = []
    accepted_route_miss_ids: list[str] = []
    doc_miss_ids: list[str] = []
    evidence_failure_ids: list[str] = []
    generation_failure_ids: list[str] = []
    doc_hit_but_low_relevancy_ids: list[str] = []
    external_guidance_ids: list[str] = []
    reasonable_refusal_ids: list[str] = []
    false_refusal_ids: list[str] = []
    unclassified_refusal_ids: list[str] = []
    unclassified_refusal_details: list[dict[str, Any]] = []
    partial_compare_ids: list[str] = []
    limited_partial_ids: list[str] = []
    covered_branch_answered_ids: list[str] = []
    no_context_guardrail_ids: list[str] = []
    empty_support_pack_guardrail_ids: list[str] = []
    zero_citation_guardrail_ids: list[str] = []
    covered_but_refused_ids: list[str] = []
    partial_answer_has_conclusion_ids: list[str] = []
    missing_branch_not_disclosed_ids: list[str] = []
    unsupported_branch_claim_ids: list[str] = []
    final_answer_mode_distribution: dict[str, int] = {}

    for item in records:
        sample_id = item["id"]
        retrieval_eval = item.get("retrieval_eval", {})
        generation_eval = item.get("generation_eval", {})
        response = str(item.get("response") or "")
        evidence = get_evidence_quality(item)
        debug = (item.get("api_response") or {}).get("debug") or {}
        final_answer_mode = str(retrieval_eval.get("final_answer_mode") or "")
        final_answer_mode_distribution[final_answer_mode] = final_answer_mode_distribution.get(final_answer_mode, 0) + 1

        if int(retrieval_eval.get("citation_count") or 0) == 0:
            zero_citation_ids.append(sample_id)
            if retrieval_eval.get("zero_citation_substantive_answer"):
                zero_citation_substantive_answer_ids.append(sample_id)
            if retrieval_eval.get("refusal_no_citation"):
                refusal_no_citation_ids.append(sample_id)
        if retrieval_eval.get("strict_route_match") is False:
            strict_route_miss_ids.append(sample_id)
        if retrieval_eval.get("accepted_route_match") is False:
            accepted_route_miss_ids.append(sample_id)

        doc_metrics = retrieval_eval.get("doc_id_metrics") or {}
        doc_hit = bool(doc_metrics.get("hit"))
        evidence_metrics = retrieval_eval.get("evidence_metrics") or {}
        evidence_coverage = evidence_metrics.get("evidence_coverage")
        if doc_metrics.get("expected"):
            if not doc_hit:
                doc_miss_ids.append(sample_id)
            elif isinstance(evidence_coverage, (int, float)) and evidence_coverage == 0:
                evidence_failure_ids.append(sample_id)
            if doc_hit and float(generation_eval.get("answer_relevancy") or 0.0) < 0.2:
                doc_hit_but_low_relevancy_ids.append(sample_id)
        if is_generation_failure(retrieval_eval, generation_eval):
            generation_failure_ids.append(sample_id)
        if retrieval_eval.get("reasonable_refusal"):
            reasonable_refusal_ids.append(sample_id)
        if retrieval_eval.get("false_refusal"):
            false_refusal_ids.append(sample_id)
        if retrieval_eval.get("unclassified_refusal"):
            unclassified_refusal_ids.append(sample_id)
            unclassified_refusal_details.append({
                "id": sample_id,
                "final_answer_mode": retrieval_eval.get("final_answer_mode"),
                "should_refuse_final": get_should_refuse_final(item),
                "refusal_reason": retrieval_eval.get("refusal_reason"),
                "support_pack_count": retrieval_eval.get("support_pack_count"),
                "candidate_support_pack_count": retrieval_eval.get("candidate_support_pack_count"),
                "covered_branch_count": retrieval_eval.get("covered_branch_count"),
                "doc_id_hit": doc_hit,
                "citation_count": retrieval_eval.get("citation_count"),
                "answer_preview": get_answer_preview(item),
            })
        if retrieval_eval.get("partial_compare"):
            partial_compare_ids.append(sample_id)
        if final_answer_mode in {"limited_partial", "limited_partial_compare"}:
            limited_partial_ids.append(sample_id)
            if has_partial_compare_conclusion(response):
                partial_answer_has_conclusion_ids.append(sample_id)
        if not retrieval_eval.get("did_refuse") and normalize_int(retrieval_eval.get("covered_branch_count")) > 0:
            covered_branch_answered_ids.append(sample_id)
        if normalize_int(debug.get("final_context_count")) == 0:
            no_context_guardrail_ids.append(sample_id)
        if evidence.get("empty_support_pack_guardrail_triggered"):
            empty_support_pack_guardrail_ids.append(sample_id)
        if evidence.get("zero_citation_guardrail_triggered"):
            zero_citation_guardrail_ids.append(sample_id)
        if retrieval_eval.get("did_refuse") and normalize_int(retrieval_eval.get("covered_branch_count")) > 0:
            allow_partial_threshold = normalize_int((item.get("dataset_meta") or {}).get("allow_partial_if_doc_coverage"))
            if allow_partial_threshold > 0 and normalize_int(retrieval_eval.get("covered_branch_count")) >= allow_partial_threshold:
                covered_but_refused_ids.append(sample_id)
        if normalize_int(retrieval_eval.get("missing_branch_count")) > 0 and normalize_int(retrieval_eval.get("missing_branch_disclosed_count")) == 0:
            missing_branch_not_disclosed_ids.append(sample_id)
        if normalize_int(retrieval_eval.get("unsupported_branch_claim_count")) > 0:
            unsupported_branch_claim_ids.append(sample_id)

        if _contains_external_guidance(response):
            external_guidance_ids.append(sample_id)

    return {
        "counts": {
            "zero_citation": len(zero_citation_ids),
            "zero_citation_substantive_answer_count": len(zero_citation_substantive_answer_ids),
            "refusal_no_citation_count": len(refusal_no_citation_ids),
            "strict_route_miss": len(strict_route_miss_ids),
            "accepted_route_miss": len(accepted_route_miss_ids),
            "doc_miss": len(doc_miss_ids),
            "evidence_failure": len(evidence_failure_ids),
            "generation_failure": len(generation_failure_ids),
            "doc_hit_but_low_relevancy": len(doc_hit_but_low_relevancy_ids),
            "external_guidance_in_answer": len(external_guidance_ids),
            "reasonable_refusal_count": len(reasonable_refusal_ids),
            "false_refusal_count": len(false_refusal_ids),
            "unclassified_refusal_count": len(unclassified_refusal_ids),
            "partial_compare_count": len(partial_compare_ids),
            "limited_partial_count": len(limited_partial_ids),
            "covered_branch_answered_count": len(covered_branch_answered_ids),
            "no_context_guardrail_triggered": len(no_context_guardrail_ids),
            "empty_support_pack_guardrail_triggered": len(empty_support_pack_guardrail_ids),
            "zero_citation_guardrail_triggered": len(zero_citation_guardrail_ids),
            "covered_but_refused_count": len(covered_but_refused_ids),
            "partial_answer_has_conclusion_count": len(partial_answer_has_conclusion_ids),
            "missing_branch_not_disclosed_count": len(missing_branch_not_disclosed_ids),
            "unsupported_branch_claim_count": len(unsupported_branch_claim_ids),
        },
        "samples": {
            "zero_citation_ids": zero_citation_ids,
            "zero_citation_substantive_answer_ids": zero_citation_substantive_answer_ids,
            "refusal_no_citation_ids": refusal_no_citation_ids,
            "strict_route_miss_ids": strict_route_miss_ids,
            "accepted_route_miss_ids": accepted_route_miss_ids,
            "doc_miss_ids": doc_miss_ids,
            "evidence_failure_ids": evidence_failure_ids,
            "generation_failure_ids": generation_failure_ids,
            "doc_hit_but_low_relevancy_ids": doc_hit_but_low_relevancy_ids,
            "external_guidance_ids": external_guidance_ids,
            "reasonable_refusal_ids": reasonable_refusal_ids,
            "false_refusal_ids": false_refusal_ids,
            "unclassified_refusal_ids": unclassified_refusal_ids,
            "unclassified_refusal_details": unclassified_refusal_details,
            "partial_compare_ids": partial_compare_ids,
            "limited_partial_ids": limited_partial_ids,
            "covered_branch_answered_ids": covered_branch_answered_ids,
            "no_context_guardrail_ids": no_context_guardrail_ids,
            "empty_support_pack_guardrail_ids": empty_support_pack_guardrail_ids,
            "zero_citation_guardrail_ids": zero_citation_guardrail_ids,
            "covered_but_refused_ids": covered_but_refused_ids,
            "partial_answer_has_conclusion_ids": partial_answer_has_conclusion_ids,
            "missing_branch_not_disclosed_ids": missing_branch_not_disclosed_ids,
            "unsupported_branch_claim_ids": unsupported_branch_claim_ids,
            "final_answer_mode_distribution": final_answer_mode_distribution,
        },
    }


def is_generation_failure(retrieval_eval: dict[str, Any], generation_eval: dict[str, Any]) -> bool:
    if not generation_eval:
        return False
    doc_metrics = retrieval_eval.get("doc_id_metrics") or {}
    evidence_metrics = retrieval_eval.get("evidence_metrics") or {}
    evidence_coverage = evidence_metrics.get("evidence_coverage")
    doc_hit = bool(doc_metrics.get("hit")) if doc_metrics.get("expected") else True
    if not doc_hit or evidence_coverage == 0:
        return False
    answer_correctness_score = generation_eval.get("answer_correctness")
    faithfulness_score = generation_eval.get("faithfulness")
    if isinstance(answer_correctness_score, (int, float)) and answer_correctness_score < 0.45:
        return True
    if isinstance(faithfulness_score, (int, float)) and faithfulness_score < 0.45:
        return True
    return False


_ABSTAIN_PATTERNS = (
    "证据不足",
    "无法可靠作答",
    "当前知识库中没有",
    "当前检索到的证据与问题直接相关性不足",
    "没有检索到可支撑",
    "缺少交叉证据",
    "无法完成完整对比",
    "证据覆盖不足",
    "文库中未提供",
    "文库中没有",
)


def _is_abstention_response(response: str) -> bool:
    if not response or not response.strip():
        return True
    return any(pattern in response for pattern in _ABSTAIN_PATTERNS)


def build_abstention_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    abstain_records = []
    non_abstain_records = []

    for item in records:
        behavior = (item.get("dataset_meta") or {}).get("expected_behavior", "")
        if behavior == "abstain_when_insufficient":
            abstain_records.append(item)
        else:
            non_abstain_records.append(item)

    if not abstain_records and not non_abstain_records:
        return {}

    abstain_details: list[dict[str, Any]] = []
    correct_abstentions = 0
    for item in abstain_records:
        response = str(item.get("response") or "")
        did_abstain = _is_abstention_response(response)
        if did_abstain:
            correct_abstentions += 1
        abstain_details.append({
            "id": item["id"],
            "did_abstain": did_abstain,
            "response_snippet": response[:120],
        })

    non_abstain_gen = []
    for item in non_abstain_records:
        gen_eval = item.get("generation_eval") or {}
        if gen_eval:
            non_abstain_gen.append(gen_eval)

    non_abstain_summary: dict[str, Any] = {}
    for metric in ("faithfulness", "answer_correctness", "answer_relevancy", "context_recall"):
        values = [
            float(g[metric])
            for g in non_abstain_gen
            if isinstance(g.get(metric), (int, float)) and not _is_nan(g[metric])
        ]
        if values:
            non_abstain_summary[metric] = round(mean(values), 4)

    result: dict[str, Any] = {
        "abstain_count": len(abstain_records),
        "non_abstain_count": len(non_abstain_records),
    }

    if abstain_records:
        result["abstention_accuracy"] = round(correct_abstentions / len(abstain_records), 4)
        result["correct_abstentions"] = correct_abstentions
        result["abstain_details"] = abstain_details

    if non_abstain_summary:
        result["non_abstain_generation"] = non_abstain_summary

    return result


def build_round8_policy_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    comparison_refuse_ids: list[str] = []
    comparison_single_doc_answered_ids: list[str] = []
    comparison_single_doc_refuse_ids: list[str] = []
    disabled_by_round8_ids: list[str] = []
    recovered_by_claim_fallback_ids: list[str] = []
    final_refuse_ids: list[str] = []
    doc_hit_but_refused_ids: list[str] = []
    claim_fallback_eligible_ids: list[str] = []
    judge_nan_ids: list[str] = []

    for item in records:
        sample_id = item["id"]
        route = str(item.get("api_response", {}).get("route") or "")
        debug = item.get("api_response", {}).get("debug") or {}
        evidence = debug.get("evidence_quality") or {}
        retrieval_eval = item.get("retrieval_eval") or {}
        doc_metrics = retrieval_eval.get("doc_id_metrics") or {}
        generation_eval = item.get("generation_eval") or {}

        is_comparison = route == "comparison" or item.get("dataset_meta", {}).get("expected_route") == "comparison"
        did_refuse = bool(retrieval_eval.get("did_refuse"))
        if did_refuse:
            final_refuse_ids.append(sample_id)
        if did_refuse and doc_metrics.get("expected") and doc_metrics.get("hit"):
            doc_hit_but_refused_ids.append(sample_id)
        if evidence.get("recovered_by_claim_fallback"):
            recovered_by_claim_fallback_ids.append(sample_id)
        if evidence.get("claim_fallback_enabled") and evidence.get("should_refuse_original"):
            claim_fallback_eligible_ids.append(sample_id)

        for metric in ("faithfulness", "answer_relevancy", "answer_correctness", "context_recall"):
            val = generation_eval.get(metric)
            if isinstance(val, float) and math.isnan(val):
                judge_nan_ids.append(sample_id)
                break

        if not is_comparison:
            continue

        if did_refuse:
            comparison_refuse_ids.append(sample_id)

        if evidence.get("hard_refusal_reason") == "comparison_single_doc":
            if evidence.get("hard_refusal_disabled_by_round8"):
                disabled_by_round8_ids.append(sample_id)
            if did_refuse:
                comparison_single_doc_refuse_ids.append(sample_id)
            else:
                comparison_single_doc_answered_ids.append(sample_id)

    recovered_by_hard_refusal_disable_count = len(disabled_by_round8_ids) - len(comparison_single_doc_refuse_ids)
    round8_total_recovered_count = recovered_by_hard_refusal_disable_count + len(recovered_by_claim_fallback_ids)
    claim_fallback_eligible_count = len(claim_fallback_eligible_ids)
    claim_fallback_recovery_rate = (
        round(len(recovered_by_claim_fallback_ids) / claim_fallback_eligible_count, 4)
        if claim_fallback_eligible_count > 0
        else None
    )

    return {
        "counts": {
            "comparison_refuse_count": len(comparison_refuse_ids),
            "comparison_single_doc_answered_count": len(comparison_single_doc_answered_ids),
            "comparison_single_doc_refuse_count": len(comparison_single_doc_refuse_ids),
            "hard_refusal_disabled_by_round8_count": len(disabled_by_round8_ids),
            "recovered_by_hard_refusal_disable_count": recovered_by_hard_refusal_disable_count,
            "recovered_by_claim_fallback_count": len(recovered_by_claim_fallback_ids),
            "round8_total_recovered_count": round8_total_recovered_count,
            "claim_fallback_eligible_count": claim_fallback_eligible_count,
            "claim_fallback_recovery_rate": claim_fallback_recovery_rate,
            "final_refuse_count": len(final_refuse_ids),
            "doc_hit_but_refused_count": len(doc_hit_but_refused_ids),
            "judge_nan_count": len(judge_nan_ids),
        },
        "samples": {
            "comparison_refuse_ids": comparison_refuse_ids,
            "comparison_single_doc_answered_ids": comparison_single_doc_answered_ids,
            "comparison_single_doc_refuse_ids": comparison_single_doc_refuse_ids,
            "hard_refusal_disabled_by_round8_ids": disabled_by_round8_ids,
            "final_refuse_ids": final_refuse_ids,
            "doc_hit_but_refused_ids": doc_hit_but_refused_ids,
            "recovered_by_claim_fallback_ids": recovered_by_claim_fallback_ids,
            "claim_fallback_eligible_ids": claim_fallback_eligible_ids,
            "judge_nan_ids": judge_nan_ids,
        },
    }


def _contains_external_guidance(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in [
            "外部补充",
            "建议调用外部工具",
            "建议触发外部工具",
            "人工复核",
            "pubmed",
            "crossref",
        ]
    )


def evaluate_generation(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    embeddings: BaseRagasEmbeddings | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float], list[str]]:
    has_reference = all(bool(item["reference"]) for item in records)
    metrics = build_metrics(has_reference=has_reference, skip_reference_metrics=args.skip_reference_metrics)
    judge_llm = build_judge_llm(args)
    judge_embeddings = embeddings or build_judge_embeddings()
    dataset = build_ragas_dataset(records)
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
        show_progress=True,
    )
    score_rows = result.to_pandas().to_dict(orient="records")
    per_item: list[dict[str, Any]] = []
    for item, row in zip(records, score_rows, strict=True):
        normalized_row = {
            slugify_metric_name(key): value
            for key, value in row.items()
            if key != "user_input"
        }
        item["generation_eval"] = normalized_row
        per_item.append({"id": item["id"], **normalized_row})
    return per_item, summarize_scores(score_rows), [metric.name for metric in metrics]


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    output_path = resolve_output_path(args.output, args.dataset)
    records = load_records(args.dataset)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    if not records:
        raise ValueError("评测集为空")

    include_debug = (
        args.include_debug
        or args.enable_round8_policy
        or args.disable_comparison_single_doc_hard_refusal
        or args.enable_claim_fallback
        or args.enable_partial_answer
        or args.enable_route_specific_thresholds
        or args.enable_comparison_prompt_v2
    )
    enriched = fetch_answers(base_url, records, args.timeout, include_debug)
    judge_embeddings = build_judge_embeddings()
    retrieval_items, retrieval_summary = evaluate_retrieval(enriched, embeddings=judge_embeddings)
    generation_items: list[dict[str, Any]] = []
    generation_summary: dict[str, float] = {}
    generation_metrics: list[str] = []
    if not args.skip_generation_metrics:
        generation_items, generation_summary, generation_metrics = evaluate_generation(
            args,
            enriched,
            embeddings=judge_embeddings,
        )

    report = {
        "base_url": base_url,
        "dataset": args.dataset,
        "judge_model": args.judge_model,
        "sample_count": len(enriched),
        "report_type": "rag_regression_report",
        "round8_flags": {
            "enable_round8_policy": args.enable_round8_policy,
            "disable_comparison_single_doc_hard_refusal": (
                args.disable_comparison_single_doc_hard_refusal
            ),
            "enable_claim_fallback": args.enable_claim_fallback,
            "enable_partial_answer": args.enable_partial_answer,
            "enable_route_specific_thresholds": args.enable_route_specific_thresholds,
            "enable_comparison_prompt_v2": args.enable_comparison_prompt_v2,
        },
        "retrieval": {
            "summary": retrieval_summary,
            "items": retrieval_items,
        },
        "generation": {
            "metrics": generation_metrics,
            "summary": generation_summary,
            "items": generation_items,
        },
        "items": enriched,
    }
    report["enterprise_slices"] = build_slice_summary(enriched, embeddings=judge_embeddings)
    report["enterprise_gates"] = build_enterprise_gates(report)
    report["diagnostics"] = build_failure_diagnostics(enriched)
    report["abstention_analysis"] = build_abstention_analysis(enriched)
    report["round8_policy_analysis"] = build_round8_policy_analysis(enriched)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    run_config_path = output_path.parent / "run_config.json"
    run_config = build_run_config(args, output_path, base_url)
    run_config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    refresh_results_index()

    print("\n[retrieval_summary]")
    print(json.dumps(retrieval_summary, ensure_ascii=False, indent=2))
    if generation_summary:
        print("\n[generation_summary]")
        print(json.dumps(generation_summary, ensure_ascii=False, indent=2))

    abstention = report["abstention_analysis"]
    if abstention:
        print("\n[abstention_analysis]")
        compact = {k: v for k, v in abstention.items() if k != "abstain_details"}
        print(json.dumps(compact, ensure_ascii=False, indent=2))
        if abstention.get("abstain_details"):
            print("  abstain_details:")
            for detail in abstention["abstain_details"]:
                status = "✓ 正确弃权" if detail["did_abstain"] else "✗ 错误生成"
                print(f"    {detail['id']}: {status}")

    print("\n[enterprise_gates]")
    print(json.dumps(report["enterprise_gates"], ensure_ascii=False, indent=2))
    print("\n[diagnostics]")
    print(json.dumps(report["diagnostics"], ensure_ascii=False, indent=2))
    print("\n[round8_policy_analysis]")
    print(json.dumps(report["round8_policy_analysis"], ensure_ascii=False, indent=2))
    print(f"\n报告已写入: {output_path}")
    print(f"运行配置已写入: {run_config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
