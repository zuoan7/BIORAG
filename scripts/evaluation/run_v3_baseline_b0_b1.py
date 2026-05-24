from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.synbio_rag import RAGApplicationService, Settings
from src.synbio_rag.domain.schemas import QueryFilters


DATASET_PATH = Path("data/eval/datasets/v3_baseline_dataset.jsonl")
RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
CHILD_CHUNK_REF_RE = re.compile(r"matched child:\s*([A-Za-z0-9_]+_sec\d+_chunk\d+::child\d+)")
REQUIRED_JUDGE_SCHEMA = {
    "answer_correctness": (
        "correctness_score",
        "correctness_pass",
        "critical_error",
        "must_include_judgments",
        "reject_if_triggered",
    ),
    "faithfulness": (
        "faithfulness_score",
        "faithfulness_pass",
        "claims",
    ),
    "citation_accuracy": (
        "citation_support_score",
        "citation_pass",
        "claim_citation_judgments",
    ),
}

VARIANTS: dict[str, dict[str, Any]] = {
    "b0_stable": {
        "display_name": "B0 稳定抽取式基线",
        "description": "核心 hybrid/BM25/source-floor/local rerank/citation；不开 parent expansion、query rewrite、LLM synthesis、table preview merge。",
        "overrides": {
            "retrieval.hybrid_enabled": True,
            "retrieval.bm25_enabled": True,
            "retrieval.source_floor_enabled": True,
            "retrieval.parent_expansion_enabled": False,
            "retrieval.rerank_mode": "plain",
            "retrieval.rerank_top_k": 10,
            "retrieval.final_top_k": 8,
            "retrieval.table_preview_enabled": True,
            "retrieval.table_preview_merge_enabled": False,
            "retrieval.table_preview_allow_formal_citation": False,
            "retrieval.original_cn_fallback_enabled": False,
            "generation.v2_profile": "stable",
            "generation.v2_use_qwen_synthesis": False,
            "generation.v2_enable_comparison_coverage": False,
            "generation.v2_enable_neighbor_audit": False,
            "generation.v2_require_citation": True,
            "generation.v2_use_history": False,
            "query_rewrite.mode": "off",
            "table_enhancement.enabled": False,
        },
    },
    "b1_parent_expansion": {
        "display_name": "B1 Parent Expansion 基线",
        "description": "在 B0 基础上只开启 parent expansion。",
        "overrides": {
            "retrieval.hybrid_enabled": True,
            "retrieval.bm25_enabled": True,
            "retrieval.source_floor_enabled": True,
            "retrieval.parent_expansion_enabled": True,
            "retrieval.parent_expansion_max_total": 12,
            "retrieval.parent_expansion_per_seed_limit": 2,
            "retrieval.rerank_mode": "plain",
            "retrieval.rerank_top_k": 10,
            "retrieval.final_top_k": 8,
            "retrieval.table_preview_enabled": True,
            "retrieval.table_preview_merge_enabled": False,
            "retrieval.table_preview_allow_formal_citation": False,
            "retrieval.original_cn_fallback_enabled": False,
            "generation.v2_profile": "stable",
            "generation.v2_use_qwen_synthesis": False,
            "generation.v2_enable_comparison_coverage": False,
            "generation.v2_enable_neighbor_audit": False,
            "generation.v2_require_citation": True,
            "generation.v2_use_history": False,
            "query_rewrite.mode": "off",
            "table_enhancement.enabled": False,
        },
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v3 B0/B1 baseline evaluation.")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--probe-concurrency", default="1,2,4,6,8")
    parser.add_argument("--probe-requests-per-level", type=int, default=8)
    parser.add_argument("--judge-concurrency", type=int, default=0)
    parser.add_argument("--judge-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()

    load_dotenv(".env")
    samples = load_jsonl(Path(args.dataset))
    if args.limit > 0:
        samples = samples[: args.limit]

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = RESULTS_ROOT / f"v3_baseline_b0_b1_{run_id}"
    report_dir = REPORTS_ROOT / f"v3_baseline_b0_b1_{run_id}"
    result_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    judge_required = not (args.skip_judge and args.skip_probe)
    judge = JudgeClient(
        timeout_seconds=args.judge_timeout_seconds,
        max_tokens=args.judge_max_tokens,
        required=judge_required,
    )
    concurrency_levels = [int(v) for v in args.probe_concurrency.split(",") if v.strip()]
    if args.skip_probe:
        probe = build_skipped_probe(
            judge=judge,
            selected_concurrency=args.judge_concurrency or 1,
        )
    else:
        probe = run_concurrency_probe(
            judge=judge,
            samples=samples,
            levels=concurrency_levels,
            requests_per_level=args.probe_requests_per_level,
        )
    write_json(result_dir / "concurrency_probe.json", probe)

    if args.probe_only:
        write_markdown(report_dir / "concurrency_probe.md", render_probe_report(probe))
        return

    selected_concurrency = args.judge_concurrency or int(probe["selected_concurrency"])
    all_summaries: dict[str, Any] = {}
    for variant_key, spec in VARIANTS.items():
        variant_result_dir = result_dir / variant_key
        variant_result_dir.mkdir(parents=True, exist_ok=True)
        results = run_variant(
            variant_key=variant_key,
            spec=spec,
            samples=samples,
            judge=judge,
            judge_concurrency=selected_concurrency,
            skip_judge=args.skip_judge,
            output_dir=variant_result_dir,
        )
        summary = summarize_variant(results)
        summary["variant_key"] = variant_key
        summary["variant_name"] = spec["display_name"]
        summary["variant_description"] = spec["description"]
        summary["config_overrides"] = spec["overrides"]
        all_summaries[variant_key] = summary
        write_json(variant_result_dir / "summary.json", summary)
        write_markdown(report_dir / f"{variant_key}_failures.md", render_failure_report(spec, results))

    comparison = compare_variants(all_summaries)
    run_config = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(Path(args.dataset)),
        "sample_count": len(samples),
        "judge_model": judge.model,
        "judge_base_url": safe_url(judge.base_url),
        "selected_judge_concurrency": selected_concurrency,
        "skip_judge": args.skip_judge,
        "skip_probe": args.skip_probe,
        "variants": {
            key: {
                "display_name": spec["display_name"],
                "description": spec["description"],
                "overrides": spec["overrides"],
            }
            for key, spec in VARIANTS.items()
        },
    }
    write_json(result_dir / "run_config.json", run_config)
    write_json(result_dir / "comparison.json", comparison)
    write_markdown(
        report_dir / "report.md",
        render_main_report(
            run_config=run_config,
            probe=probe,
            summaries=all_summaries,
            comparison=comparison,
            result_dir=result_dir,
        ),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


class JudgeClient:
    def __init__(self, *, timeout_seconds: float, max_tokens: int, required: bool = True) -> None:
        self.base_url = (os.getenv("DEEPSEEK_URL") or "").strip()
        self.api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
        self.model = (os.getenv("DEEPSEEK_MODEL_NAME") or "").strip()
        missing = [
            name
            for name, value in {
                "DEEPSEEK_URL": self.base_url,
                "DEEPSEEK_API_KEY": self.api_key,
                "DEEPSEEK_MODEL_NAME": self.model,
            }.items()
            if not value
        ]
        if missing and required:
            raise RuntimeError(f"Missing DeepSeek env vars: {', '.join(missing)}")
        if missing:
            self.base_url = self.base_url or "not_configured"
            self.model = self.model or "not_configured"
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens

    def complete_json(self, payload: dict[str, Any], *, purpose: str) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install openai to run judge-enabled evaluation.") from exc

        prompt = build_judge_prompt(payload, purpose=purpose)
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "你是严谨的 RAG 评测裁判。只输出合法 JSON，不要输出 Markdown。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(extract_json_object(content))
        parsed["_judge_usage"] = usage_to_dict(getattr(response, "usage", None))
        parsed["_judge_finish_reason"] = getattr(response.choices[0], "finish_reason", None)
        return parsed


def build_judge_prompt(payload: dict[str, Any], *, purpose: str) -> str:
    return (
        "请按给定 rubric 对 RAG 输出进行评估。评价文字必须使用中文。字段名必须保持英文。\n"
        "不要引入外部知识，只能依据输入中的 rubric、参考答案、系统答案和引用上下文。\n"
        "正确率按 answer_rubric.must_include / acceptable_variants / reject_if 判断。\n"
        "忠实度按 system_answer 中事实性 claim 是否被 citations 的 quote 支持判断。\n"
        "引用准确率按 answer 中的事实是否被对应 citations 支持判断；如果没有 citation，则 citation_support_score 为 0。\n"
        "若 reject_if 被触发，correctness_pass 必须为 false。\n"
        "若答案存在明显编造、数值方向错误、显著性反转或把不同实体混淆，critical_error 必须为 true。\n"
        "输出 JSON schema：\n"
        "{\n"
        '  "answer_correctness": {\n'
        '    "must_include_judgments": [{"item": "...", "score": 0.0, "matched_text": "", "reason": "..."}],\n'
        '    "reject_if_triggered": [{"item": "...", "reason": "..."}],\n'
        '    "numeric_errors": ["..."],\n'
        '    "conclusion_errors": ["..."],\n'
        '    "critical_error": false,\n'
        '    "correctness_score": 0.0,\n'
        '    "correctness_pass": false,\n'
        '    "reason": "..."\n'
        "  },\n"
        '  "faithfulness": {\n'
        '    "claims": [{"claim": "...", "support_status": "supported|unsupported|contradicted|not_factual", "citation_ids": [1], "reason": "..."}],\n'
        '    "unsupported_claims": ["..."],\n'
        '    "contradicted_claims": ["..."],\n'
        '    "faithfulness_score": 0.0,\n'
        '    "faithfulness_pass": false,\n'
        '    "reason": "..."\n'
        "  },\n"
        '  "citation_accuracy": {\n'
        '    "claim_citation_judgments": [{"claim": "...", "citation_ids": [1], "support_status": "supported|unsupported|wrong_citation|missing_citation", "reason": "..."}],\n'
        '    "unsupported_cited_claims": ["..."],\n'
        '    "wrong_citation_claims": ["..."],\n'
        '    "citation_support_score": 0.0,\n'
        '    "citation_pass": false,\n'
        '    "reason": "..."\n'
        "  },\n"
        '  "overall": {"judge_notes": "..."}\n'
        "}\n"
        f"purpose: {purpose}\n"
        "输入：\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge output: {text[:200]}")
    return match.group(0)


def usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def run_concurrency_probe(
    *,
    judge: JudgeClient,
    samples: list[dict[str, Any]],
    levels: list[int],
    requests_per_level: int,
) -> dict[str, Any]:
    probe_samples = [samples[i % len(samples)] for i in range(max(requests_per_level, 1))]
    level_results = []
    for level in levels:
        started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
            futures = [
                executor.submit(run_probe_request, judge, sample, idx)
                for idx, sample in enumerate(probe_samples)
            ]
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())
        elapsed = time.perf_counter() - started
        successes = [row for row in rows if row["ok"]]
        latencies = [row["latency_seconds"] for row in rows if row["ok"]]
        level_results.append(
            {
                "concurrency": level,
                "request_count": len(rows),
                "success_count": len(successes),
                "error_count": len(rows) - len(successes),
                "success_rate": round(len(successes) / len(rows), 4) if rows else 0.0,
                "elapsed_seconds": round(elapsed, 3),
                "latency_p50_seconds": percentile(latencies, 50),
                "latency_p95_seconds": percentile(latencies, 95),
                "max_latency_seconds": round(max(latencies), 3) if latencies else None,
                "errors": [row["error"] for row in rows if not row["ok"]][:5],
            }
        )
    selected = select_concurrency(level_results)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "judge_model": judge.model,
        "judge_base_url": safe_url(judge.base_url),
        "requests_per_level": requests_per_level,
        "levels": level_results,
        "selected_concurrency": selected,
        "selection_policy": "选择成功率 100%、p95 <= 30 秒的最高并发；若没有满足条件，选择成功率最高且 p95 最低的并发。",
    }


def build_skipped_probe(*, judge: JudgeClient, selected_concurrency: int) -> dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "judge_model": judge.model,
        "judge_base_url": safe_url(judge.base_url),
        "requests_per_level": 0,
        "levels": [],
        "selected_concurrency": selected_concurrency,
        "selection_policy": "本轮通过 --skip-probe 跳过并发探测，使用显式 judge 并发或默认值。",
        "skipped": True,
    }


def run_probe_request(judge: JudgeClient, sample: dict[str, Any], idx: int) -> dict[str, Any]:
    started = time.perf_counter()
    payload = {
        "sample_id": f"probe_{idx}_{sample.get('sample_id', '')}",
        "question": sample.get("question", ""),
        "expected_answer": sample.get("expected_answer", ""),
        "answer_rubric": sample.get("answer_rubric", {}),
        "system_answer": sample.get("expected_answer", ""),
        "citations": [
            {
                "citation_id": 1,
                "doc_id": (sample.get("expected_doc_ids") or [""])[0] if sample.get("expected_doc_ids") else "",
                "chunk_id": sample.get("target_chunk_id_candidate") or "",
                "quote": sample.get("expected_answer", ""),
            }
        ],
    }
    try:
        result = judge.complete_json(payload, purpose="并发探测")
        schema_errors = validate_judge_schema(result)
        ok = not schema_errors
        return {
            "ok": ok,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "error": "" if ok else "; ".join(schema_errors),
        }
    except Exception as exc:
        return {
            "ok": False,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "error": scrub_error(str(exc), judge.api_key),
        }


def select_concurrency(levels: list[dict[str, Any]]) -> int:
    passing = [
        item
        for item in levels
        if item["success_rate"] == 1.0
        and item["latency_p95_seconds"] is not None
        and item["latency_p95_seconds"] <= 30
    ]
    if passing:
        return int(max(passing, key=lambda item: item["concurrency"])["concurrency"])
    best = sorted(
        levels,
        key=lambda item: (
            -float(item["success_rate"]),
            float(item["latency_p95_seconds"] or 9999),
            int(item["concurrency"]),
        ),
    )[0]
    return int(best["concurrency"])


def run_variant(
    *,
    variant_key: str,
    spec: dict[str, Any],
    samples: list[dict[str, Any]],
    judge: JudgeClient,
    judge_concurrency: int,
    skip_judge: bool,
    output_dir: Path,
) -> list[dict[str, Any]]:
    settings = Settings.from_env()
    apply_overrides(settings, spec["overrides"])
    patch_transformers_prepare_for_model()
    service = RAGApplicationService(settings)

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for idx, sample in enumerate(samples, start=1):
        item_started = time.perf_counter()
        response = service.ask(
            sample["question"],
            filters=QueryFilters(),
            include_debug=True,
        )
        row = build_eval_row(
            variant_key=variant_key,
            sample=sample,
            response=response,
            latency_seconds=time.perf_counter() - item_started,
        )
        rows.append(row)
        if idx % 10 == 0:
            write_jsonl(output_dir / "results.partial.jsonl", rows)
            print(
                f"[{variant_key}] RAG completed {idx}/{len(samples)}; elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    if not skip_judge:
        rows = run_judges(
            rows=rows,
            judge=judge,
            concurrency=judge_concurrency,
            output_path=output_dir / "results.partial.jsonl",
        )

    write_jsonl(output_dir / "results.jsonl", rows)
    partial = output_dir / "results.partial.jsonl"
    if partial.exists():
        partial.unlink()
    return rows


def patch_transformers_prepare_for_model() -> None:
    try:
        from transformers.models.xlm_roberta.tokenization_xlm_roberta import (
            XLMRobertaTokenizer,
        )
    except Exception:
        return
    if hasattr(XLMRobertaTokenizer, "prepare_for_model"):
        return

    def prepare_for_model(
        self,
        ids,
        pair_ids=None,
        truncation=False,
        max_length=None,
        padding=False,
        **_kwargs,
    ):
        first = list(ids or [])
        second = list(pair_ids or []) if pair_ids is not None else None
        if max_length is not None and second is not None:
            special_count = self.num_special_tokens_to_add(pair=True)
            allowed_second = max(0, int(max_length) - len(first) - special_count)
            if truncation in {"only_second", True} and len(second) > allowed_second:
                second = second[:allowed_second]
        if second is None:
            input_ids = [self.cls_token_id] + first + [self.sep_token_id]
        else:
            input_ids = [
                self.cls_token_id,
                *first,
                self.sep_token_id,
                self.sep_token_id,
                *second,
                self.sep_token_id,
            ]
        token_type_ids = [self.pad_token_type_id] * len(input_ids)
        return {"input_ids": input_ids, "token_type_ids": token_type_ids}

    XLMRobertaTokenizer.prepare_for_model = prepare_for_model


def apply_overrides(settings: Settings, overrides: dict[str, Any]) -> None:
    for dotted, value in overrides.items():
        target: Any = settings
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)


def build_eval_row(
    *,
    variant_key: str,
    sample: dict[str, Any],
    response: Any,
    latency_seconds: float,
) -> dict[str, Any]:
    response_dict = dataclass_to_dict(response)
    debug = response_dict.get("debug") or {}
    retrieval_output = debug.get("retrieval_output") or {}
    evidence_lifecycle_debug = debug.get("evidence_lifecycle_debug") or {}
    rerank_output = (debug.get("evidence_lifecycle_debug") or {}).get("rerank_output") or {}
    seed_output = evidence_lifecycle_debug.get("seed_chunks") or debug.get("seed_output") or {}
    final_output = evidence_lifecycle_debug.get("final_chunks") or debug.get("final_output") or {}
    raw_retrieved_chunk_ids = list(retrieval_output.get("chunk_ids") or [])
    raw_retrieved_doc_ids = list(retrieval_output.get("doc_ids") or [])
    retrieved_doc_ids = list(rerank_output.get("doc_ids") or [])[:10]
    retrieved_chunk_ids = list(rerank_output.get("chunk_ids") or [])[:10]
    retrieved_parent_chunk_ids = dedupe([parent_chunk_id(chunk_id) for chunk_id in retrieved_chunk_ids])
    seed_chunk_ids = list(seed_output.get("chunk_ids") or [])
    seed_parent_chunk_ids = dedupe([parent_chunk_id(chunk_id) for chunk_id in seed_chunk_ids])
    final_chunk_ids = list(final_output.get("kept_chunk_ids") or final_output.get("chunk_ids") or [])
    final_parent_chunk_ids = dedupe([parent_chunk_id(chunk_id) for chunk_id in final_chunk_ids])
    generation_debug = debug.get("generation_v2") or {}
    support_chunk_ids = extract_support_chunk_ids(generation_debug)
    support_matched_child_chunk_ids = extract_support_matched_child_ids(generation_debug)
    citation_matched_child_chunk_ids = extract_citation_matched_child_ids(generation_debug)
    citations = response_dict.get("citations") or []
    citation_doc_ids = [citation.get("doc_id", "") for citation in citations]
    citation_chunk_ids = [citation.get("chunk_id", "") for citation in citations]
    citation_parent_chunk_ids = dedupe([parent_chunk_id(chunk_id) for chunk_id in citation_chunk_ids])
    inferred_matched_child_chunk_ids = extract_inferred_matched_child_ids(
        response_dict.get("answer", ""),
        citations,
    )
    gold_chunk_ids = get_gold_chunk_ids(sample)
    gold_parent_chunk_ids = dedupe([parent_chunk_id(chunk_id) for chunk_id in gold_chunk_ids])
    metrics = compute_rule_metrics(
        sample=sample,
        retrieved_doc_ids=retrieved_doc_ids,
        retrieved_chunk_ids=retrieved_chunk_ids,
        support_chunk_ids=support_chunk_ids,
        support_matched_child_chunk_ids=support_matched_child_chunk_ids,
        citations=citations,
        citation_matched_child_chunk_ids=citation_matched_child_chunk_ids,
        inferred_matched_child_chunk_ids=inferred_matched_child_chunk_ids,
        actual_route=response_dict.get("route", ""),
    )
    return {
        "variant_key": variant_key,
        "sample_id": sample.get("sample_id"),
        "question": sample.get("question"),
        "expected_route": sample.get("expected_route"),
        "actual_route": str(response_dict.get("route", "")),
        "category": sample.get("category"),
        "difficulty": sample.get("difficulty"),
        "expected_doc_ids": sample.get("expected_doc_ids") or [],
        "expected_source_files": sample.get("expected_source_files") or [],
        "expected_answer": sample.get("expected_answer"),
        "answer_rubric": sample.get("answer_rubric"),
        "target_chunk_id_candidate": sample.get("target_chunk_id_candidate"),
        "stable_target_block_ids": sample.get("stable_target_block_ids") or [],
        "gold_chunk_ids": gold_chunk_ids,
        "gold_parent_chunk_ids": gold_parent_chunk_ids,
        "system_answer": response_dict.get("answer", ""),
        "confidence": response_dict.get("confidence"),
        "latency_seconds": round(latency_seconds, 3),
        "retrieved_doc_ids_top10": retrieved_doc_ids,
        "retrieved_chunk_ids_top10": retrieved_chunk_ids,
        "retrieved_parent_chunk_ids_top10": retrieved_parent_chunk_ids,
        "raw_retrieved_doc_ids": raw_retrieved_doc_ids,
        "raw_retrieved_chunk_ids": raw_retrieved_chunk_ids,
        "raw_retrieved_parent_chunk_ids": dedupe([parent_chunk_id(chunk_id) for chunk_id in raw_retrieved_chunk_ids]),
        "seed_chunk_ids": seed_chunk_ids,
        "seed_parent_chunk_ids": seed_parent_chunk_ids,
        "final_chunk_ids": final_chunk_ids,
        "final_parent_chunk_ids": final_parent_chunk_ids,
        "support_chunk_ids": support_chunk_ids,
        "support_matched_child_chunk_ids": support_matched_child_chunk_ids,
        "citation_doc_ids": citation_doc_ids,
        "citation_chunk_ids": citation_chunk_ids,
        "citation_parent_chunk_ids": citation_parent_chunk_ids,
        "citation_matched_child_chunk_ids": citation_matched_child_chunk_ids,
        "inferred_matched_child_chunk_ids": inferred_matched_child_chunk_ids,
        "citations": [
            {
                "citation_id": idx,
                "doc_id": citation.get("doc_id", ""),
                "chunk_id": citation.get("chunk_id", ""),
                "source_file": citation.get("source_file", ""),
                "section": citation.get("section", ""),
                "quote": citation.get("quote", ""),
            }
            for idx, citation in enumerate(citations, start=1)
        ],
        "rule_metrics": metrics,
        "debug_digest": {
            "retrieved_count": debug.get("retrieved_count"),
            "reranked_count": debug.get("reranked_count"),
            "seed_context_count": debug.get("seed_context_count"),
            "final_context_count": debug.get("final_context_count"),
            "latency_ms": debug.get("latency_ms"),
            "query_bundle": debug.get("query_bundle") or {},
            "table_preview": debug.get("table_preview"),
            "retrieval_output": compact_stage_debug(retrieval_output),
            "raw_child_trace": compact_raw_child_trace(debug.get("filter_strategy") or {}),
            "rerank_output": compact_stage_debug(rerank_output),
            "seed_chunks": compact_stage_debug(seed_output),
            "final_chunks": compact_stage_debug(final_output),
            "rerank_hits": compact_reranker_debug(debug.get("rerank_hits") or {}),
            "parent_expansion": compact_parent_expansion_debug(debug.get("parent_expansion") or {}),
            "generation_v2": compact_generation_debug(debug.get("generation_v2") or {}),
        },
    }


def dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: dataclass_to_dict(val) for key, val in asdict(value).items()}
    if isinstance(value, list):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: dataclass_to_dict(val) for key, val in value.items()}
    return value


def get_gold_chunk_ids(sample: dict[str, Any]) -> list[str]:
    source_trace = ((sample.get("answer_rubric") or {}).get("source_trace") or {})
    gold_chunks = [str(v) for v in source_trace.get("chunk_ids") or []]
    if sample.get("target_chunk_id_candidate"):
        gold_chunks.append(str(sample["target_chunk_id_candidate"]))
    return dedupe(gold_chunks)


def parent_chunk_id(chunk_id: Any) -> str:
    return str(chunk_id or "").split("::child", 1)[0]


def extract_support_chunk_ids(generation_debug: dict[str, Any]) -> list[str]:
    support_pack = generation_debug.get("support_pack") or []
    if not isinstance(support_pack, list):
        return []
    return dedupe([item.get("chunk_id", "") for item in support_pack if isinstance(item, dict)])


def extract_support_matched_child_ids(generation_debug: dict[str, Any]) -> list[str]:
    support_pack = generation_debug.get("support_pack") or []
    child_ids: list[str] = []
    if not isinstance(support_pack, list):
        return child_ids
    for item in support_pack:
        if not isinstance(item, dict):
            continue
        child_ids.extend(str(v) for v in item.get("matched_child_chunk_ids") or [])
    return dedupe(child_ids)


def extract_citation_matched_child_ids(generation_debug: dict[str, Any]) -> list[str]:
    binding = nested(generation_debug, "support_selection_debug", "citation_binding")
    if not isinstance(binding, dict):
        return []
    ordered_evidence_ids = binding.get("ordered_evidence_ids") or []
    child_ids_by_evidence_id = binding.get("matched_child_chunk_ids_by_evidence_id") or {}
    if not isinstance(ordered_evidence_ids, list) or not isinstance(child_ids_by_evidence_id, dict):
        return []
    child_ids: list[str] = []
    for evidence_id in ordered_evidence_ids:
        child_ids.extend(str(v) for v in child_ids_by_evidence_id.get(str(evidence_id), []) or [])
    return dedupe(child_ids)


def extract_inferred_matched_child_ids(answer: Any, citations: list[dict[str, Any]]) -> list[str]:
    text_parts = [str(answer or "")]
    for citation in citations:
        text_parts.append(str(citation.get("quote") or ""))
    return dedupe(CHILD_CHUNK_REF_RE.findall("\n".join(text_parts)))


def compute_rule_metrics(
    *,
    sample: dict[str, Any],
    retrieved_doc_ids: list[str],
    retrieved_chunk_ids: list[str],
    support_chunk_ids: list[str],
    support_matched_child_chunk_ids: list[str],
    citations: list[dict[str, Any]],
    citation_matched_child_chunk_ids: list[str],
    inferred_matched_child_chunk_ids: list[str],
    actual_route: str,
) -> dict[str, Any]:
    expected_docs = [str(v) for v in sample.get("expected_doc_ids") or []]
    gold_chunks = get_gold_chunk_ids(sample)
    gold_parent_chunks = dedupe([parent_chunk_id(chunk) for chunk in gold_chunks])
    retrieved_docs_set = set(retrieved_doc_ids)
    retrieved_chunks_set = set(retrieved_chunk_ids)
    retrieved_parent_chunks = dedupe([parent_chunk_id(chunk) for chunk in retrieved_chunk_ids])
    retrieved_parent_chunks_set = set(retrieved_parent_chunks)
    support_parent_chunks_set = {parent_chunk_id(chunk) for chunk in support_chunk_ids}
    support_matched_child_chunks_set = set(support_matched_child_chunk_ids)
    cited_docs = [str(citation.get("doc_id", "")) for citation in citations]
    cited_chunks = [str(citation.get("chunk_id", "")) for citation in citations]
    cited_parent_chunks_set = {parent_chunk_id(chunk) for chunk in cited_chunks}
    citation_matched_child_chunks_set = set(citation_matched_child_chunk_ids)
    inferred_matched_child_chunks_set = set(inferred_matched_child_chunk_ids)
    expected_route = str(sample.get("expected_route") or "")
    is_negative = expected_route == "negative"

    doc_hits = [doc for doc in expected_docs if doc in retrieved_docs_set]
    doc_recall = len(doc_hits) / len(expected_docs) if expected_docs else None
    doc_all_hit = bool(expected_docs) and len(doc_hits) == len(expected_docs)
    doc_any_hit = bool(doc_hits) if expected_docs else None
    doc_mrr = reciprocal_rank(retrieved_doc_ids, set(expected_docs)) if expected_docs else None
    chunk_hits = [chunk for chunk in gold_chunks if chunk in retrieved_chunks_set]
    chunk_recall = len(chunk_hits) / len(gold_chunks) if gold_chunks else None
    chunk_mrr = reciprocal_rank(retrieved_chunk_ids, set(gold_chunks)) if gold_chunks else None
    parent_chunk_hits = [chunk for chunk in gold_parent_chunks if chunk in retrieved_parent_chunks_set]
    parent_chunk_recall = len(parent_chunk_hits) / len(gold_parent_chunks) if gold_parent_chunks else None
    parent_chunk_mrr = reciprocal_rank(retrieved_parent_chunks, set(gold_parent_chunks)) if gold_parent_chunks else None
    support_parent_chunk_hits = [chunk for chunk in gold_parent_chunks if chunk in support_parent_chunks_set]
    support_child_evidence_hits = [chunk for chunk in gold_chunks if chunk in support_matched_child_chunks_set]
    citation_doc_hits = [doc for doc in expected_docs if doc in set(cited_docs)]
    citation_chunk_hits = [chunk for chunk in gold_chunks if chunk in set(cited_chunks)]
    citation_parent_chunk_hits = [chunk for chunk in gold_parent_chunks if chunk in cited_parent_chunks_set]
    citation_child_evidence_hits = [chunk for chunk in gold_chunks if chunk in citation_matched_child_chunks_set]
    inferred_child_evidence_hits = [chunk for chunk in gold_chunks if chunk in inferred_matched_child_chunks_set]
    route_match = normalize_route(actual_route) == expected_route

    return {
        "doc_recall_at10": round(doc_recall, 6) if doc_recall is not None else None,
        "doc_hit_at10": doc_any_hit,
        "strict_doc_all_hit_at10": doc_all_hit if expected_docs else None,
        "doc_mrr_at10": round(doc_mrr, 6) if doc_mrr is not None else None,
        "exact_gold_chunk_recall_at10": round(chunk_recall, 6) if chunk_recall is not None else None,
        "exact_gold_chunk_hit_at10": bool(chunk_hits) if gold_chunks else None,
        "exact_gold_chunk_mrr_at10": round(chunk_mrr, 6) if chunk_mrr is not None else None,
        "chunk_recall_at10": round(chunk_recall, 6) if chunk_recall is not None else None,
        "chunk_hit_at10": bool(chunk_hits) if gold_chunks else None,
        "chunk_mrr_at10": round(chunk_mrr, 6) if chunk_mrr is not None else None,
        "parent_chunk_recall_at10": round(parent_chunk_recall, 6) if parent_chunk_recall is not None else None,
        "parent_chunk_hit_at10": bool(parent_chunk_hits) if gold_parent_chunks else None,
        "parent_chunk_mrr_at10": round(parent_chunk_mrr, 6) if parent_chunk_mrr is not None else None,
        "route_match": route_match,
        "citation_doc_hit": bool(citation_doc_hits) if expected_docs and not is_negative else None,
        "citation_doc_all_hit": len(citation_doc_hits) == len(expected_docs) if expected_docs and not is_negative else None,
        "citation_chunk_hit": bool(citation_chunk_hits) if gold_chunks and not is_negative else None,
        "citation_parent_chunk_hit": bool(citation_parent_chunk_hits) if gold_parent_chunks and not is_negative else None,
        "citation_parent_chunk_all_hit": (
            len(citation_parent_chunk_hits) == len(gold_parent_chunks)
            if gold_parent_chunks and not is_negative
            else None
        ),
        "support_parent_chunk_hit": bool(support_parent_chunk_hits) if gold_parent_chunks else None,
        "support_child_evidence_hit": bool(support_child_evidence_hits) if gold_chunks else None,
        "citation_child_evidence_hit": bool(citation_child_evidence_hits) if gold_chunks and not is_negative else None,
        "inferred_child_evidence_hit": bool(inferred_child_evidence_hits) if gold_chunks else None,
        "citation_count": len(citations),
        "negative_no_citation": len(citations) == 0 if is_negative else None,
    }


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    for idx, value in enumerate(ranked[:10], start=1):
        if value in gold:
            return 1.0 / idx
    return 0.0


def normalize_route(route: str) -> str:
    value = str(route).lower().strip()
    if "." in value:
        value = value.split(".")[-1]
    return value


def run_judges(
    *,
    rows: list[dict[str, Any]],
    judge: JudgeClient,
    concurrency: int,
    output_path: Path,
) -> list[dict[str, Any]]:
    indexed = {idx: row for idx, row in enumerate(rows)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(judge_row_with_retry, judge, row): idx
            for idx, row in indexed.items()
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            rows[idx]["judge"] = future.result()
            completed += 1
            if completed % 10 == 0:
                write_jsonl(output_path, rows)
                print(f"[judge] completed {completed}/{len(rows)}", flush=True)
    return rows


def judge_row_with_retry(judge: JudgeClient, row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "sample_id": row["sample_id"],
        "question": row["question"],
        "expected_answer": row["expected_answer"],
        "answer_rubric": row["answer_rubric"],
        "system_answer": row["system_answer"],
        "citations": row["citations"],
        "expected_route": row["expected_route"],
        "category": row["category"],
    }
    for attempt in range(1, 4):
        started = time.perf_counter()
        try:
            result = judge.complete_json(payload, purpose="v3 baseline 正式评测")
            schema_errors = validate_judge_schema(result)
            if schema_errors:
                raise ValueError("judge_output_schema_invalid: " + "; ".join(schema_errors))
            result["_judge_latency_seconds"] = round(time.perf_counter() - started, 3)
            result["_judge_attempt"] = attempt
            return result
        except Exception as exc:
            last_error = scrub_error(str(exc), judge.api_key)
            if attempt < 3:
                time.sleep(1.5 * attempt)
                continue
            return {
                "judge_error": last_error,
                "_judge_latency_seconds": round(time.perf_counter() - started, 3),
                "_judge_attempt": attempt,
                "answer_correctness": {
                    "correctness_score": None,
                    "correctness_pass": False,
                    "critical_error": True,
                    "reason": f"裁判调用失败：{last_error}",
                    "must_include_judgments": [],
                    "reject_if_triggered": [],
                    "numeric_errors": [],
                    "conclusion_errors": [],
                },
                "faithfulness": {
                    "faithfulness_score": None,
                    "faithfulness_pass": False,
                    "claims": [],
                    "unsupported_claims": [],
                    "contradicted_claims": [],
                    "reason": f"裁判调用失败：{last_error}",
                },
                "citation_accuracy": {
                    "citation_support_score": None,
                    "citation_pass": False,
                    "claim_citation_judgments": [],
                    "unsupported_cited_claims": [],
                    "wrong_citation_claims": [],
                    "reason": f"裁判调用失败：{last_error}",
                },
            }


def validate_judge_schema(result: Any) -> list[str]:
    if not isinstance(result, dict):
        return ["judge output is not a JSON object"]
    errors: list[str] = []
    for section, required_keys in REQUIRED_JUDGE_SCHEMA.items():
        value = result.get(section)
        if not isinstance(value, dict):
            errors.append(f"missing or invalid section: {section}")
            continue
        for key in required_keys:
            if key not in value:
                errors.append(f"missing field: {section}.{key}")
    if errors and result.get("_judge_finish_reason"):
        errors.append(f"finish_reason={result.get('_judge_finish_reason')}")
    return errors


def summarize_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rule = [row["rule_metrics"] for row in rows]
    judges = [row.get("judge") or {} for row in rows]
    summary = {
        "sample_count": len(rows),
        "rule_metrics": {
            "doc_recall_at10_avg": avg([m.get("doc_recall_at10") for m in rule]),
            "doc_hit_at10_rate": rate([m.get("doc_hit_at10") for m in rule]),
            "strict_doc_all_hit_at10_rate": rate([m.get("strict_doc_all_hit_at10") for m in rule]),
            "doc_mrr_at10_avg": avg([m.get("doc_mrr_at10") for m in rule]),
            "exact_gold_chunk_recall_at10_avg": avg([m.get("exact_gold_chunk_recall_at10") for m in rule]),
            "exact_gold_chunk_hit_at10_rate": rate([m.get("exact_gold_chunk_hit_at10") for m in rule]),
            "exact_gold_chunk_mrr_at10_avg": avg([m.get("exact_gold_chunk_mrr_at10") for m in rule]),
            "chunk_recall_at10_avg": avg([m.get("chunk_recall_at10") for m in rule]),
            "chunk_hit_at10_rate": rate([m.get("chunk_hit_at10") for m in rule]),
            "chunk_mrr_at10_avg": avg([m.get("chunk_mrr_at10") for m in rule]),
            "parent_chunk_recall_at10_avg": avg([m.get("parent_chunk_recall_at10") for m in rule]),
            "parent_chunk_hit_at10_rate": rate([m.get("parent_chunk_hit_at10") for m in rule]),
            "parent_chunk_mrr_at10_avg": avg([m.get("parent_chunk_mrr_at10") for m in rule]),
            "route_match_rate": rate([m.get("route_match") for m in rule]),
            "citation_doc_hit_rate": rate([m.get("citation_doc_hit") for m in rule]),
            "citation_doc_all_hit_rate": rate([m.get("citation_doc_all_hit") for m in rule]),
            "citation_chunk_hit_rate": rate([m.get("citation_chunk_hit") for m in rule]),
            "citation_parent_chunk_hit_rate": rate([m.get("citation_parent_chunk_hit") for m in rule]),
            "citation_parent_chunk_all_hit_rate": rate([m.get("citation_parent_chunk_all_hit") for m in rule]),
            "support_parent_chunk_hit_rate": rate([m.get("support_parent_chunk_hit") for m in rule]),
            "support_child_evidence_hit_rate": rate([m.get("support_child_evidence_hit") for m in rule]),
            "citation_child_evidence_hit_rate": rate([m.get("citation_child_evidence_hit") for m in rule]),
            "inferred_child_evidence_hit_rate": rate([m.get("inferred_child_evidence_hit") for m in rule]),
            "negative_no_citation_rate": rate([m.get("negative_no_citation") for m in rule]),
        },
        "judge_metrics": {
            "answer_correctness_score_avg": avg(
                [nested(j, "answer_correctness", "correctness_score") for j in judges]
            ),
            "answer_correctness_pass_rate": rate(
                [nested(j, "answer_correctness", "correctness_pass") for j in judges]
            ),
            "critical_error_rate": rate(
                [nested(j, "answer_correctness", "critical_error") for j in judges]
            ),
            "faithfulness_score_avg": avg(
                [nested(j, "faithfulness", "faithfulness_score") for j in judges]
            ),
            "faithfulness_pass_rate": rate(
                [nested(j, "faithfulness", "faithfulness_pass") for j in judges]
            ),
            "citation_support_score_avg": avg(
                [nested(j, "citation_accuracy", "citation_support_score") for j in judges]
            ),
            "citation_pass_rate": rate(
                [nested(j, "citation_accuracy", "citation_pass") for j in judges]
            ),
            "judge_error_count": sum(1 for j in judges if j.get("judge_error")),
            "judge_success_count": sum(1 for j in judges if j and not j.get("judge_error")),
        },
        "latency": {
            "rag_latency_avg_seconds": avg([row.get("latency_seconds") for row in rows]),
            "rag_latency_p95_seconds": percentile([row.get("latency_seconds") for row in rows], 95),
            "judge_latency_avg_seconds": avg([j.get("_judge_latency_seconds") for j in judges]),
            "judge_latency_p95_seconds": percentile([j.get("_judge_latency_seconds") for j in judges], 95),
        },
        "slices": build_slices(rows),
        "diagnostic_buckets": build_diagnostic_buckets(rows),
        "top_failures": build_top_failures(rows, limit=30),
    }
    return summary


def build_diagnostic_buckets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = {
        "doc_miss": {
            "label": "doc miss",
            "description": "期望文档未进入 rerank top10。",
            "sample_ids": [],
        },
        "doc_hit_parent_chunk_miss": {
            "label": "doc hit 但 parent chunk miss",
            "description": "期望文档命中，但期望父块未进入 rerank top10。",
            "sample_ids": [],
        },
        "parent_hit_support_miss": {
            "label": "parent hit 但 support miss",
            "description": "期望父块进入 rerank top10，但未进入 generation support。",
            "sample_ids": [],
        },
        "parent_hit_citation_miss": {
            "label": "parent hit 但 citation miss",
            "description": "期望父块进入 rerank top10，但未进入最终引用。",
            "sample_ids": [],
        },
        "evidence_hit_answer_failed": {
            "label": "evidence 命中但答案失败",
            "description": "support 或 citation 命中期望父块/子证据，但 judge 判定答案正确性未通过。",
            "sample_ids": [],
        },
        "judge_schema_or_call_issue": {
            "label": "judge/schema 问题",
            "description": "judge 调用失败或输出缺少必需 schema。",
            "sample_ids": [],
        },
    }
    by_category: dict[str, dict[str, int]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        category = str(row.get("category") or "unknown")
        by_category.setdefault(category, {key: 0 for key in buckets})
        rule = row.get("rule_metrics") or {}
        judge = row.get("judge") or {}
        schema_errors = validate_judge_schema(judge) if judge else []
        bucket_keys: list[str] = []
        if rule.get("doc_hit_at10") is False:
            bucket_keys.append("doc_miss")
        if rule.get("doc_hit_at10") is True and rule.get("parent_chunk_hit_at10") is False:
            bucket_keys.append("doc_hit_parent_chunk_miss")
        if rule.get("parent_chunk_hit_at10") is True and rule.get("support_parent_chunk_hit") is False:
            bucket_keys.append("parent_hit_support_miss")
        if rule.get("parent_chunk_hit_at10") is True and rule.get("citation_parent_chunk_hit") is False:
            bucket_keys.append("parent_hit_citation_miss")

        evidence_hit = any(
            rule.get(key) is True
            for key in (
                "support_parent_chunk_hit",
                "support_child_evidence_hit",
                "citation_parent_chunk_hit",
                "citation_child_evidence_hit",
            )
        )
        if evidence_hit and nested(judge, "answer_correctness", "correctness_pass") is False:
            bucket_keys.append("evidence_hit_answer_failed")
        if judge.get("judge_error") or schema_errors:
            bucket_keys.append("judge_schema_or_call_issue")

        for key in bucket_keys:
            buckets[key]["sample_ids"].append(sample_id)
            by_category[category][key] += 1

    sample_count = len(rows)
    return {
        key: {
            "label": value["label"],
            "description": value["description"],
            "count": len(value["sample_ids"]),
            "rate": round(len(value["sample_ids"]) / sample_count, 6) if sample_count else None,
            "sample_ids": value["sample_ids"],
            "sample_ids_preview": value["sample_ids"][:30],
        }
        for key, value in buckets.items()
    } | {"by_category": by_category}


def build_slices(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("expected_route", "category", "difficulty"):
        result[key] = {}
        values = sorted({str(row.get(key) or "unknown") for row in rows})
        for value in values:
            subset = [row for row in rows if str(row.get(key) or "unknown") == value]
            result[key][value] = {
                "sample_count": len(subset),
                "doc_recall_at10_avg": avg([row["rule_metrics"].get("doc_recall_at10") for row in subset]),
                "doc_hit_at10_rate": rate([row["rule_metrics"].get("doc_hit_at10") for row in subset]),
                "parent_chunk_hit_at10_rate": rate(
                    [row["rule_metrics"].get("parent_chunk_hit_at10") for row in subset]
                ),
                "citation_parent_chunk_hit_rate": rate(
                    [row["rule_metrics"].get("citation_parent_chunk_hit") for row in subset]
                ),
                "citation_child_evidence_hit_rate": rate(
                    [row["rule_metrics"].get("citation_child_evidence_hit") for row in subset]
                ),
                "answer_correctness_pass_rate": rate(
                    [nested(row.get("judge") or {}, "answer_correctness", "correctness_pass") for row in subset]
                ),
                "faithfulness_pass_rate": rate(
                    [nested(row.get("judge") or {}, "faithfulness", "faithfulness_pass") for row in subset]
                ),
                "citation_pass_rate": rate(
                    [nested(row.get("judge") or {}, "citation_accuracy", "citation_pass") for row in subset]
                ),
            }
    return result


def build_top_failures(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    failures = []
    for row in rows:
        judge = row.get("judge") or {}
        rule = row.get("rule_metrics") or {}
        reasons = []
        if rule.get("doc_hit_at10") is False:
            reasons.append("检索未命中期望文档")
        elif rule.get("parent_chunk_hit_at10") is False:
            reasons.append("命中期望文档但未命中期望父块")
        elif rule.get("citation_parent_chunk_hit") is False:
            reasons.append("父块进入候选但未进入最终引用")
        if nested(judge, "answer_correctness", "correctness_pass") is False:
            reasons.append(nested(judge, "answer_correctness", "reason") or "答案正确率未通过")
        if nested(judge, "faithfulness", "faithfulness_pass") is False:
            reasons.append(nested(judge, "faithfulness", "reason") or "忠实度未通过")
        if nested(judge, "citation_accuracy", "citation_pass") is False:
            reasons.append(nested(judge, "citation_accuracy", "reason") or "引用准确率未通过")
        if not reasons:
            continue
        failures.append(
            {
                "sample_id": row["sample_id"],
                "category": row["category"],
                "expected_route": row["expected_route"],
                "doc_recall_at10": rule.get("doc_recall_at10"),
                "parent_chunk_hit_at10": rule.get("parent_chunk_hit_at10"),
                "citation_parent_chunk_hit": rule.get("citation_parent_chunk_hit"),
                "correctness_score": nested(judge, "answer_correctness", "correctness_score"),
                "faithfulness_score": nested(judge, "faithfulness", "faithfulness_score"),
                "citation_support_score": nested(judge, "citation_accuracy", "citation_support_score"),
                "reasons": reasons[:3],
            }
        )
    failures.sort(
        key=lambda item: (
            item["doc_recall_at10"] or 0,
            item["correctness_score"] if item["correctness_score"] is not None else -1,
            item["faithfulness_score"] if item["faithfulness_score"] is not None else -1,
        )
    )
    return failures[:limit]


def compare_variants(summaries: dict[str, Any]) -> dict[str, Any]:
    if "b0_stable" not in summaries or "b1_parent_expansion" not in summaries:
        return {}
    b0 = flatten_summary_metrics(summaries["b0_stable"])
    b1 = flatten_summary_metrics(summaries["b1_parent_expansion"])
    deltas = {}
    for key, value in b1.items():
        if key in b0 and isinstance(value, (int, float)) and isinstance(b0[key], (int, float)):
            deltas[key] = round(value - b0[key], 6)
    return {
        "baseline": "b0_stable",
        "candidate": "b1_parent_expansion",
        "deltas": deltas,
        "interpretation": "正值表示 B1 高于 B0；负值表示 B1 低于 B0。",
    }


def flatten_summary_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    flat = {}
    for section in ("rule_metrics", "judge_metrics", "latency"):
        for key, value in (summary.get(section) or {}).items():
            flat[f"{section}.{key}"] = value
    return flat


def render_main_report(
    *,
    run_config: dict[str, Any],
    probe: dict[str, Any],
    summaries: dict[str, Any],
    comparison: dict[str, Any],
    result_dir: Path,
) -> str:
    lines = [
        "# v3 baseline B0/B1 评测报告",
        "",
        f"- 运行 ID：`{run_config['run_id']}`",
        f"- 数据集：`{run_config['dataset_path']}`",
        f"- 样本数：{run_config['sample_count']}",
        f"- Judge 模型：`{run_config['judge_model']}`",
        f"- Judge Base URL：`{run_config['judge_base_url']}`",
        f"- 选定 Judge 并发：{run_config['selected_judge_concurrency']}",
        f"- 结果目录：`{result_dir}`",
        "",
        "## 评估口径",
        "",
        "- 检索指标同时统计 doc、父 chunk、exact gold chunk 三个层级；父 chunk 会将 `::childNNN` 归一到所属父块。",
        "- 子证据指标使用系统 debug 中的 `matched_child_chunk_ids`；若历史结果缺少该字段，可另行用答案/quote 文本推断，但正式报告优先使用显式 debug 字段。",
        "- 答案正确率、忠实度和引用准确率由 DeepSeek judge 按样本 rubric、参考答案、系统答案和引用上下文打分；字段名保持英文，评价文字使用中文。",
        "- 答案通过率使用 judge 的 `correctness_pass`；连续分使用 `correctness_score`、`faithfulness_score`、`citation_support_score`。",
        "",
        "## 并发探测",
        "",
        "| 并发 | 请求数 | 成功率 | p50(s) | p95(s) | 错误数 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in probe["levels"]:
        lines.append(
            f"| {item['concurrency']} | {item['request_count']} | {item['success_rate']:.2%} | "
            f"{fmt(item['latency_p50_seconds'])} | {fmt(item['latency_p95_seconds'])} | {item['error_count']} |"
        )
    if probe.get("skipped"):
        lines.append("| N/A | 0 | N/A | N/A | N/A | 0 |")
        lines.append("")
        lines.append("本轮通过 `--skip-probe` 跳过并发探测。")
    lines.extend(["", f"选定并发：**{probe['selected_concurrency']}**。", ""])
    lines.extend(["## 总体指标", ""])
    lines.append(
        "| 变体 | doc recall@10 | doc MRR@10 | parent hit@10 | parent MRR@10 | exact chunk hit@10 | citation parent hit | citation child hit | 答案通过率 | 忠实度通过率 | 引用通过率 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, summary in summaries.items():
        lines.append(
            f"| {summary['variant_name']} | "
            f"{fmt(summary['rule_metrics']['doc_recall_at10_avg'])} | "
            f"{fmt(summary['rule_metrics']['doc_mrr_at10_avg'])} | "
            f"{pct(summary['rule_metrics']['parent_chunk_hit_at10_rate'])} | "
            f"{fmt(summary['rule_metrics']['parent_chunk_mrr_at10_avg'])} | "
            f"{pct(summary['rule_metrics']['exact_gold_chunk_hit_at10_rate'])} | "
            f"{pct(summary['rule_metrics']['citation_parent_chunk_hit_rate'])} | "
            f"{pct(summary['rule_metrics']['citation_child_evidence_hit_rate'])} | "
            f"{pct(summary['judge_metrics']['answer_correctness_pass_rate'])} | "
            f"{pct(summary['judge_metrics']['faithfulness_pass_rate'])} | "
            f"{pct(summary['judge_metrics']['citation_pass_rate'])} |"
        )
    if comparison:
        lines.extend(["", "## B1 相对 B0 差值", ""])
        lines.append("| 指标 | 差值 |")
        lines.append("|---|---:|")
        for key, value in sorted(comparison.get("deltas", {}).items()):
            lines.append(f"| `{key}` | {fmt(value)} |")
    lines.extend(["", "## 诊断桶", ""])
    lines.append("以下诊断桶允许重叠，用于定位问题发生在哪个环节。")
    lines.append("")
    for variant_key, summary in summaries.items():
        lines.append(f"### {summary['variant_name']}")
        lines.append("")
        lines.append("| 问题类型 | 样本数 | 占比 | 样本预览 |")
        lines.append("|---|---:|---:|---|")
        for bucket_key, item in (summary.get("diagnostic_buckets") or {}).items():
            if bucket_key == "by_category":
                continue
            preview = ", ".join(item.get("sample_ids_preview") or [])
            lines.append(
                f"| {item.get('label', bucket_key)} | {item.get('count', 0)} | "
                f"{pct(item.get('rate'))} | {preview} |"
            )
        lines.append("")
    lines.extend(["", "## 按类别切片", ""])
    for variant_key, summary in summaries.items():
        lines.append(f"### {summary['variant_name']}")
        lines.append("")
        lines.append("| 类别 | 样本数 | doc hit@10 | parent hit@10 | citation parent hit | citation child hit | 答案通过率 | 忠实度通过率 | 引用通过率 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for category, item in (summary["slices"]["category"] or {}).items():
            lines.append(
                f"| {category} | {item['sample_count']} | {pct(item['doc_hit_at10_rate'])} | "
                f"{pct(item['parent_chunk_hit_at10_rate'])} | "
                f"{pct(item['citation_parent_chunk_hit_rate'])} | "
                f"{pct(item['citation_child_evidence_hit_rate'])} | "
                f"{pct(item['answer_correctness_pass_rate'])} | "
                f"{pct(item['faithfulness_pass_rate'])} | {pct(item['citation_pass_rate'])} |"
            )
        lines.append("")
    lines.extend(["## 配置说明", ""])
    for key, variant in run_config["variants"].items():
        lines.append(f"### {variant['display_name']}")
        lines.append("")
        lines.append(variant["description"])
        lines.append("")
        lines.append("| 开关 | 值 |")
        lines.append("|---|---|")
        for name, value in variant["overrides"].items():
            lines.append(f"| `{name}` | `{value}` |")
        lines.append("")
    return "\n".join(lines)


def render_probe_report(probe: dict[str, Any]) -> str:
    lines = [
        "# DeepSeek Judge 并发探测报告",
        "",
        f"- 模型：`{probe['judge_model']}`",
        f"- Base URL：`{probe['judge_base_url']}`",
        f"- 每档请求数：{probe['requests_per_level']}",
        f"- 选定并发：{probe['selected_concurrency']}",
        "",
        "| 并发 | 请求数 | 成功率 | p50(s) | p95(s) | 错误数 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in probe["levels"]:
        lines.append(
            f"| {item['concurrency']} | {item['request_count']} | {item['success_rate']:.2%} | "
            f"{fmt(item['latency_p50_seconds'])} | {fmt(item['latency_p95_seconds'])} | {item['error_count']} |"
        )
    if probe.get("skipped"):
        lines.append("| N/A | 0 | N/A | N/A | N/A | 0 |")
        lines.append("")
        lines.append("本轮通过 `--skip-probe` 跳过并发探测。")
    return "\n".join(lines)


def render_failure_report(spec: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    failures = build_top_failures(rows, limit=80)
    lines = [
        f"# {spec['display_name']} 失败样本摘要",
        "",
        spec["description"],
        "",
        "| sample_id | route | category | doc recall@10 | parent hit@10 | citation parent hit | 正确率 | 忠实度 | 引用支持 | 原因 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in failures:
        lines.append(
            f"| {item['sample_id']} | {item['expected_route']} | {item['category']} | "
            f"{fmt(item['doc_recall_at10'])} | {fmt(item['parent_chunk_hit_at10'])} | "
            f"{fmt(item['citation_parent_chunk_hit'])} | {fmt(item['correctness_score'])} | "
            f"{fmt(item['faithfulness_score'])} | {fmt(item['citation_support_score'])} | "
            f"{'<br>'.join(item['reasons'])} |"
        )
    return "\n".join(lines)


def compact_stage_debug(debug: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "input_count",
        "output_count",
        "chunk_ids",
        "kept_chunk_ids",
        "cited_chunk_ids",
        "dropped_chunk_ids",
        "uncited_selected_support_chunk_ids",
        "doc_ids",
        "cited_doc_ids",
        "evidence_ids",
        "kept_evidence_ids",
        "parent_chunk_ids",
        "matched_child_chunk_ids",
        "matched_child_chunk_ids_by_chunk_id",
        "matched_child_chunk_ids_by_evidence_id",
        "drop_reasons",
        "ordered_evidence_ids",
        "invalid_evidence_ids",
        "protected_seed_kept_count",
        "protected_seed_dropped_count",
        "protected_seed_count",
        "protected_seed_chunk_ids",
        "protected_seed_kept_chunk_ids",
        "protected_seed_dropped_chunk_ids",
    ]
    return {key: debug.get(key) for key in keys if key in debug}


def compact_raw_child_trace(debug: dict[str, Any]) -> dict[str, Any]:
    trace = debug.get("raw_child_trace") or []
    if not isinstance(trace, list):
        trace = []
    aggregation = debug.get("child_to_parent_aggregation") or {}
    if not isinstance(aggregation, dict):
        aggregation = {}
    kept_trace: list[dict[str, Any]] = []
    for item in trace:
        if not isinstance(item, dict):
            continue
        kept_trace.append(
            {
                "rank": item.get("rank"),
                "child_chunk_id": item.get("child_chunk_id"),
                "parent_chunk_id": item.get("parent_chunk_id"),
                "doc_id": item.get("doc_id"),
                "source_file": item.get("source_file"),
                "section": item.get("section"),
                "source": item.get("source"),
                "vector_score": round_number(item.get("vector_score")),
                "bm25_score": round_number(item.get("bm25_score")),
                "fusion_score": round_number(item.get("fusion_score")),
                "chunk_index": item.get("chunk_index"),
                "child_index": item.get("child_index"),
            }
        )
    return {
        "raw_child_trace": kept_trace,
        "child_to_parent_aggregation": {
            "raw_child_count": aggregation.get("raw_child_count"),
            "raw_parent_count": aggregation.get("raw_parent_count"),
            "materialized_parent_count": aggregation.get("materialized_parent_count"),
            "parent_ids_from_children": list(aggregation.get("parent_ids_from_children") or []),
            "child_ids_by_parent_id": {
                str(parent_id): [str(child_id) for child_id in child_ids or []]
                for parent_id, child_ids in (aggregation.get("child_ids_by_parent_id") or {}).items()
            },
        },
    }


def compact_parent_expansion_debug(debug: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "enabled",
        "reason",
        "input_count",
        "output_count",
        "strategy",
        "effective_intent",
        "effective_max_total",
        "effective_per_seed_limit",
        "effective_max_added",
        "effective_final_context_cap",
        "limit_reason",
        "seed_preservation_enabled",
        "added_chunk_ids",
        "added_parent_ids",
        "added_parent_types",
        "per_seed_added",
        "per_doc_added",
        "protected_seed_count",
        "protected_seed_chunk_ids",
        "final_contains_rerank_top3",
        "primary_doc_window_gating",
        "window_target_doc_id",
        "window_gating_reason",
        "primary_doc_local_context_gating",
        "local_context_target_doc_id",
        "local_context_gating_reason",
        "page_candidates_found",
        "page_candidates_added",
        "evidence_candidates_found",
        "evidence_candidates_added",
        "caption_context_candidates_found",
        "caption_context_added",
        "summary_sections_added",
        "page_skipped_reason",
        "evidence_skipped_reason",
    ]
    return {key: debug.get(key) for key in keys if key in debug}


def compact_generation_debug(debug: dict[str, Any]) -> dict[str, Any]:
    support_pack = debug.get("support_pack") or []
    citation_child_ids = extract_citation_matched_child_ids(debug)
    candidates = debug.get("candidates") or []
    support_selection = debug.get("support_selection_debug") or {}
    citation_binding = (
        support_selection.get("citation_binding")
        if isinstance(support_selection, dict)
        else {}
    ) or {}
    return {
        "candidate_count": debug.get("candidate_count"),
        "answer_mode": debug.get("answer_mode"),
        "plan_mode": (debug.get("answer_plan") or {}).get("mode") if isinstance(debug.get("answer_plan"), dict) else None,
        "citation_count": debug.get("citation_count"),
        "support_pack_size": len(support_pack) if isinstance(support_pack, list) else None,
        "support_matched_child_count": len(extract_support_matched_child_ids(debug)),
        "citation_matched_child_count": len(citation_child_ids),
        "citation_output": debug.get("citation_output"),
        "candidates": compact_candidates(candidates),
        "support_selector": compact_support_selector(debug),
        "support_pack": compact_support_pack(support_pack),
        "citation_candidates": compact_citation_candidates(citation_binding),
        "citation_binding": compact_citation_binding(citation_binding),
        "evidence_lifecycle": compact_generation_lifecycle(
            debug.get("evidence_lifecycle_debug") or {}
        ),
    }


def compact_reranker_debug(debug: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(debug, dict):
        return {}
    selection = debug.get("selection") if isinstance(debug.get("selection"), dict) else {}
    return {
        "mode": debug.get("mode"),
        "original_query": debug.get("original_query"),
        "rewritten_query": debug.get("rewritten_query"),
        "rerank_query": debug.get("rerank_query"),
        "query_variants": list(debug.get("query_variants") or []),
        "query_scores": debug.get("query_scores") or [],
        "selection": {
            "mode": selection.get("mode"),
            "top_k": selection.get("top_k"),
            "pre_floor_chunk_ids": selection.get("pre_floor_chunk_ids") or [],
            "post_floor_chunk_ids": selection.get("post_floor_chunk_ids") or [],
            "score_floor": selection.get("score_floor") or {},
            "doc_diversity": selection.get("doc_diversity") or {},
            "comparison_selection": selection.get("comparison_selection") or {},
            "pre_body_coverage_final_chunk_ids": selection.get("pre_body_coverage_final_chunk_ids") or [],
            "same_doc_body_coverage": selection.get("same_doc_body_coverage") or {},
            "final_chunk_ids": selection.get("final_chunk_ids") or [],
            "final_doc_ids": selection.get("final_doc_ids") or [],
        },
        "ranking_trace": debug.get("ranking_trace") or [],
        "final_hits": debug.get("final_hits") or [],
    }


def compact_candidates(candidates: Any) -> list[dict[str, Any]]:
    if not isinstance(candidates, list):
        return []
    return [compact_candidate(candidate) for candidate in candidates if isinstance(candidate, dict)]


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    chunk_id = str(candidate.get("chunk_id") or "")
    return {
        "evidence_id": candidate.get("evidence_id"),
        "chunk_id": chunk_id,
        "parent_chunk_id": parent_chunk_id(chunk_id),
        "doc_id": candidate.get("doc_id"),
        "source_file": candidate.get("source_file"),
        "section": candidate.get("section"),
        "rerank_rank": metadata.get("rerank_rank"),
        "scores": {
            "vector_score": round_number(candidate.get("vector_score")),
            "bm25_score": round_number(candidate.get("bm25_score")),
            "rerank_score": round_number(candidate.get("rerank_score")),
            "fusion_score": round_number(candidate.get("fusion_score")),
        },
        "features": candidate.get("features") or {},
        "reasons": list(candidate.get("reasons") or []),
        "matched_child_chunk_ids": matched_child_ids_from_metadata(metadata),
        "parent_child_generation_view_used": bool(
            metadata.get("parent_child_generation_view_used")
        ),
        "generation_evidence_role": metadata.get("generation_evidence_role", ""),
        "text_preview": clip_text(candidate.get("text"), 240),
    }


def compact_support_selector(debug: dict[str, Any]) -> dict[str, Any]:
    support_selection = debug.get("support_selection_debug") or {}
    if not isinstance(support_selection, dict):
        support_selection = {}
    selection_debug = support_selection.get("selection_debug") or {}
    if not isinstance(selection_debug, dict):
        selection_debug = {}
    return {
        "candidate_ids": support_selection.get("candidate_ids") or [],
        "selected_evidence_ids": support_selection.get("selected_evidence_ids") or [],
        "selection_debug": {
            "intent": selection_debug.get("intent"),
            "candidate_count": selection_debug.get("candidate_count"),
            "scored_count": selection_debug.get("scored_count"),
            "eligible_count": selection_debug.get("eligible_count"),
            "below_min_score_count": selection_debug.get("below_min_score_count"),
            "selected_before_protection_evidence_ids": selection_debug.get(
                "selected_before_protection_evidence_ids"
            )
            or [],
            "selected_evidence_ids": selection_debug.get("selected_evidence_ids") or [],
            "protected_seed_evidence_ids": selection_debug.get("protected_seed_evidence_ids") or [],
            "protected_seed_inserted_evidence_ids": selection_debug.get(
                "protected_seed_inserted_evidence_ids"
            )
            or [],
            "drop_reasons_by_evidence_id": selection_debug.get("drop_reasons_by_evidence_id")
            or {},
            "support_score_ranking": selection_debug.get("support_score_ranking") or [],
        },
        "summary_selection": support_selection.get("summary_selection") or {},
    }


def compact_support_pack(support_pack: Any) -> list[dict[str, Any]]:
    if not isinstance(support_pack, list):
        return []
    rows = []
    for item in support_pack:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "")
        rows.append(
            {
                "evidence_id": item.get("evidence_id"),
                "chunk_id": chunk_id,
                "parent_chunk_id": parent_chunk_id(chunk_id),
                "doc_id": item.get("doc_id"),
                "section": item.get("section"),
                "support_score": round_number(item.get("support_score")),
                "reasons": list(item.get("reasons") or []),
                "matched_child_chunk_ids": [
                    str(value) for value in item.get("matched_child_chunk_ids") or []
                ],
                "parent_child_generation_view_used": bool(
                    item.get("parent_child_generation_view_used")
                ),
                "generation_evidence_role": item.get("generation_evidence_role", ""),
            }
        )
    return rows


def compact_citation_candidates(citation_binding: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = citation_binding.get("citation_candidates") or []
    if not isinstance(candidates, list):
        return []
    rows = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        chunk_id = str(candidate.get("chunk_id") or "")
        rows.append(
            {
                "evidence_id": candidate.get("evidence_id"),
                "chunk_id": chunk_id,
                "parent_chunk_id": parent_chunk_id(chunk_id),
                "doc_id": candidate.get("doc_id"),
                "section": candidate.get("section"),
                "rerank_rank": candidate.get("rerank_rank"),
                "support_score": round_number(candidate.get("support_score")),
                "support_priority": round_number(candidate.get("support_priority")),
                "citation_priority": round_number(candidate.get("citation_priority")),
                "citation_eligible": candidate.get("citation_eligible"),
                "drop_reason": candidate.get("drop_reason", ""),
                "is_protected_seed": candidate.get("is_protected_seed"),
                "matched_child_chunk_ids": [
                    str(value) for value in candidate.get("matched_child_chunk_ids") or []
                ],
                "parent_child_generation_view_used": bool(
                    candidate.get("parent_child_generation_view_used")
                ),
                "generation_evidence_role": candidate.get("generation_evidence_role", ""),
                "reasons": list(candidate.get("reasons") or []),
                "text_preview": clip_text(candidate.get("text"), 200),
            }
        )
    return rows


def compact_citation_binding(citation_binding: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(citation_binding, dict):
        return {}
    keys = [
        "ordered_evidence_ids",
        "invalid_evidence_ids",
        "blocked_evidence_ids",
        "input_evidence_ids",
        "uncited_selected_support_evidence_ids",
        "drop_reasons_by_evidence_id",
        "citation_candidate_count",
        "citation_eligible_count",
        "citation_completion_count",
        "plan_mode",
        "parent_child_generation_view_used_by_evidence_id",
        "matched_child_chunk_ids_by_evidence_id",
        "generation_evidence_role_by_evidence_id",
    ]
    return {key: citation_binding.get(key) for key in keys if key in citation_binding}


def compact_generation_lifecycle(debug: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(debug, dict):
        return {}
    return {
        "support_input": compact_stage_debug(debug.get("support_input") or {}),
        "selected_support": compact_stage_debug(debug.get("selected_support") or {}),
        "citation_candidates": compact_stage_debug(debug.get("citation_candidates") or {}),
        "citation_output": compact_stage_debug(debug.get("citation_output") or {}),
    }


def matched_child_ids_from_metadata(metadata: dict[str, Any]) -> list[str]:
    value = metadata.get("matched_child_chunk_ids")
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    value = metadata.get("matched_child_chunk_id")
    if value:
        return [str(value)]
    return []


def clip_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def round_number(value: Any) -> float | None:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def avg(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and v is not None]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 6)


def rate(values: list[Any]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(sum(1 for v in vals if bool(v)) / len(vals), 6)


def percentile(values: list[Any], p: int) -> float | None:
    nums = sorted(float(v) for v in values if isinstance(v, (int, float)) and v is not None)
    if not nums:
        return None
    if len(nums) == 1:
        return round(nums[0], 3)
    k = (len(nums) - 1) * p / 100
    lower = int(k)
    upper = min(lower + 1, len(nums) - 1)
    weight = k - lower
    value = nums[lower] * (1 - weight) + nums[upper] * weight
    return round(value, 3)


def nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def dedupe(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def safe_url(value: str) -> str:
    return re.sub(r"//[^/@]+@", "//<redacted>@", value).rstrip("/")


def scrub_error(text: str, api_key: str) -> str:
    value = text.replace(api_key, "<redacted-api-key>") if api_key else text
    return re.sub(r"(Bearer\s+)[A-Za-z0-9._-]+", r"\1<redacted>", value)


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


if __name__ == "__main__":
    main()
