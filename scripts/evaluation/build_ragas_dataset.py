#!/usr/bin/env python3
"""
将 smoke100 / evaluation 输出转换为 RAGAS 可用的 dataset (JSONL)。

两种 context 模式:
  --context-source final_chunks   使用进入 generation 的全部 candidate chunks text
  --context-source cited_chunks   仅使用最终 citation 引用的 chunks text

输出: results/ragas/smoke100_ragas_dataset_<mode>.jsonl

使用方式:
  python scripts/evaluation/build_ragas_dataset.py
  python scripts/evaluation/build_ragas_dataset.py --context-source cited_chunks
  python scripts/evaluation/build_ragas_dataset.py --base-url http://127.0.0.1:9000 --timeout 300
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", message=r"Importing .* from 'ragas.metrics' is deprecated.*",
                        category=DeprecationWarning)

import httpx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "data/eval/datasets/enterprise_ragas_eval_v1.json"
OUTPUT_DIR = ROOT / "results/ragas"

_SECTION_TO_GROUP: dict[str, str] = {
    "Title": "TITLE", "Abstract": "ABSTRACT",
    "Introduction": "INTRO", "Background": "INTRO",
    "Results": "RESULT", "Results and Discussion": "RESULT",
    "Discussion": "DISCUSSION", "Conclusion": "CONCLUSION",
    "Conclusions": "CONCLUSION", "Methods": "METHOD",
    "Materials and Methods": "METHOD", "Experimental Section": "METHOD",
    "Experimental Procedures": "METHOD", "Full Text": "BODY_ANY",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build RAGAS dataset from smoke100 outputs")
    p.add_argument("--dataset", default=str(DATASET_PATH),
                   help="Path to evaluation dataset JSON")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR),
                   help="Output directory for JSONL files")
    p.add_argument("--base-url", default="http://127.0.0.1:9000",
                   help="RAG API base URL")
    p.add_argument("--context-source", default="final_chunks",
                   choices=["final_chunks", "cited_chunks"],
                   help="final_chunks: all candidate chunks; cited_chunks: citation-only chunks")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="HTTP request timeout in seconds")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Max samples to process (0 = all)")
    p.add_argument("--max-context-chars", type=int, default=12000,
                   help="Max total characters for concatenated contexts per sample")
    p.add_argument("--include-debug", action="store_true",
                   help="Request include_debug=true from API for final_chunks")
    return p.parse_args()


def load_dataset(path: str) -> list[dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset not found: {path_obj}")
    with path_obj.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("samples", "items", "data"):
            val = data.get(key)
            if isinstance(val, list):
                return val
    raise ValueError("Unsupported dataset format")


def normalize_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _section_to_group(section: str) -> str:
    return _SECTION_TO_GROUP.get(section, "UNKNOWN")


def compute_section_norm_hit(expected_sections: list[str],
                             expected_groups: list[str],
                             actual_sections: list[str]) -> bool:
    """Group-level section hit (primary metric, matches evaluate_ragas.py)."""
    if not expected_sections:
        return True  # no expected sections → not applicable, treat as pass
    actual_groups = {_section_to_group(s) for s in actual_sections if s}
    exp_groups = set(expected_groups)
    return bool(exp_groups & actual_groups)


def compute_doc_id_hit(expected_doc_ids: list[str],
                       actual_doc_ids: list[str]) -> bool:
    if not expected_doc_ids:
        return True
    return bool(set(expected_doc_ids) & set(actual_doc_ids))


def build_cited_contexts(citations: list[dict[str, Any]]) -> list[str]:
    """Build context strings from citation quotes (cited_chunks mode)."""
    contexts: list[str] = []
    for c in citations:
        quote = (c.get("quote") or "").strip()
        title = (c.get("title") or "").strip()
        section = (c.get("section") or "").strip()
        source_file = (c.get("source_file") or "").strip()
        text = quote or title
        if section:
            text = f"{section}: {text}" if text else section
        if source_file:
            text = f"{source_file} | {text}" if text else source_file
        if text:
            contexts.append(text)
    return contexts


def build_final_contexts(debug: dict[str, Any]) -> list[str]:
    """Build context strings from generation_v2 candidates (final_chunks mode)."""
    gv2 = (debug or {}).get("generation_v2") or {}
    candidates = gv2.get("candidates") or []
    contexts: list[str] = []
    for c in candidates:
        text = (c.get("text") or c.get("chunk_text") or "").strip()
        if not text:
            continue
        section = (c.get("section") or "").strip()
        doc_id = (c.get("doc_id") or "").strip()
        if section:
            text = f"[{doc_id} | {section}] {text}" if doc_id else f"[{section}] {text}"
        elif doc_id:
            text = f"[{doc_id}] {text}"
        contexts.append(text)
    return contexts


def truncate_contexts(contexts: list[str], max_chars: int) -> list[str]:
    """Truncate contexts to stay within max_chars total."""
    truncated: list[str] = []
    total = 0
    for ctx in contexts:
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(ctx) > remaining:
            ctx = ctx[:remaining - 100] + "..."
        truncated.append(ctx)
        total += len(ctx)
    return truncated


def fetch_answers(base_url: str, records: list[dict[str, Any]],
                  timeout: float, include_debug: bool) -> list[dict[str, Any]]:
    """Call RAG API for each sample, return enriched records."""
    enriched: list[dict[str, Any]] = []
    with httpx.Client(timeout=timeout) as client:
        # health check
        healthz = client.get(f"{base_url}/healthz")
        healthz.raise_for_status()
        print(f"[healthz] {healthz.json()}")

        for idx, item in enumerate(records, start=1):
            question = (item.get("question") or "").strip()
            if not question:
                print(f"[skip] sample {idx}: empty question")
                continue

            payload: dict[str, Any] = {
                "question": question,
                "session_id": item.get("session_id", f"ragas_{idx:03d}"),
                "tenant_id": item.get("tenant_id", "default"),
                "doc_ids": item.get("doc_ids") or [],
                "sections": item.get("sections") or [],
                "source_files": item.get("source_files") or [],
                "min_score": item.get("min_score"),
                "include_debug": include_debug,
            }

            try:
                resp = client.post(f"{base_url}/v1/ask", json=payload)
                resp.raise_for_status()
                result = resp.json()
            except Exception as exc:
                print(f"[ERROR] sample {item.get('id', idx)}: {exc}")
                enriched.append({
                    "id": item.get("id", f"sample_{idx:03d}"),
                    "question": question,
                    "error": str(exc),
                })
                continue

            enriched.append({
                "id": item["id"],
                "question": question,
                "response": result.get("answer", ""),
                "reference": (item.get("reference") or "").strip(),
                "scenario": str(item.get("scenario") or "").strip(),
                "expected_behavior": str(item.get("expected_behavior") or "").strip(),
                "expected_route": str(item.get("expected_route") or "").strip(),
                "actual_route": str(result.get("route") or "").strip(),
                "confidence": result.get("confidence"),
                "expected_doc_ids": normalize_list(item.get("expected_doc_ids") or item.get("doc_ids")),
                "expected_sections": normalize_list(item.get("expected_sections")),
                "expected_section_groups": normalize_list(item.get("expected_section_groups")),
                "api_response": result,
            })
            print(f"[{idx}/{len(records)}] {item['id']}: route={result.get('route')} "
                  f"citations={len(result.get('citations') or [])}")

    return enriched


def build_ragas_records(enriched: list[dict[str, Any]],
                        context_source: str,
                        max_context_chars: int) -> list[dict[str, Any]]:
    """Convert enriched records to RAGAS-compatible JSONL format."""
    ragas_records: list[dict[str, Any]] = []
    for item in enriched:
        if "error" in item:
            # preserve failed samples
            ragas_records.append({
                "sample_id": item["id"],
                "question": item["question"],
                "answer": "",
                "contexts": [],
                "reference": "",
                "route": "",
                "scenario": "",
                "expected_behavior": "",
                "expected_doc_ids": [],
                "expected_section_groups": [],
                "citation_count": 0,
                "doc_id_hit": False,
                "section_norm_hit": False,
                "empty_context": True,
                "answer_mode": "error",
                "error": item["error"],
            })
            continue

        api_resp = item.get("api_response") or {}
        citations = api_resp.get("citations") or []
        debug = api_resp.get("debug") or {}

        # Build contexts based on mode
        if context_source == "final_chunks":
            contexts = build_final_contexts(debug)
            if not contexts:
                # Fallback to cited chunks if final_chunks unavailable
                contexts = build_cited_contexts(citations)
        else:
            contexts = build_cited_contexts(citations)

        contexts = truncate_contexts(contexts, max_context_chars)

        # Derive project metrics
        actual_doc_ids = [str(c.get("doc_id") or "").strip()
                         for c in citations if c.get("doc_id")]
        actual_sections = [str(c.get("section") or "").strip()
                          for c in citations if c.get("section")]

        doc_id_hit = compute_doc_id_hit(item["expected_doc_ids"], actual_doc_ids)
        section_norm_hit = compute_section_norm_hit(
            item["expected_sections"], item["expected_section_groups"], actual_sections
        )
        citation_count = len(citations)

        # Answer mode classification
        answer = item["response"]
        if not answer or not answer.strip():
            answer_mode = "empty"
        elif _is_abstention(answer):
            answer_mode = "refusal"
        else:
            answer_mode = "substantive"

        ragas_records.append({
            "sample_id": item["id"],
            "question": item["question"],
            "answer": answer,
            "contexts": contexts,
            "reference": item["reference"],
            "route": item["actual_route"],
            "scenario": item.get("scenario", ""),
            "expected_behavior": item.get("expected_behavior", ""),
            "expected_doc_ids": item.get("expected_doc_ids", []),
            "expected_section_groups": item.get("expected_section_groups", []),
            "citation_count": citation_count,
            "doc_id_hit": doc_id_hit,
            "section_norm_hit": section_norm_hit,
            "empty_context": len(contexts) == 0,
            "answer_mode": answer_mode,
            "confidence": item.get("confidence"),
        })

    return ragas_records


_ABSTAIN_PATTERNS = (
    "证据不足，无法",
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

# Weak patterns that only count as refusal if answer has NO claims structure
_REFUSAL_WEAK_PATTERNS = (
    "证据不足",
    "不能视为完整综述",
)


def _is_abstention(response: str) -> bool:
    """Check if answer is a genuine refusal (no substantive claims).

    Strong patterns always count as refusal.
    Weak patterns (e.g. in evidence-limitations caveats) only count if answer
    has no claim structure (bullet points with citations).
    """
    if not response or not response.strip():
        return True
    # Strong refusal signals
    if any(p in response for p in _ABSTAIN_PATTERNS):
        return True
    # Weak patterns: only if no structured claims present
    has_claims = "- " in response and "[" in response
    if not has_claims:
        return any(p in response for p in _REFUSAL_WEAK_PATTERNS)
    return False


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    empty_ctx = sum(1 for r in records if r.get("empty_context"))
    refusals = sum(1 for r in records if r.get("answer_mode") == "refusal")
    empty_ans = sum(1 for r in records if r.get("answer_mode") == "empty")
    errors = sum(1 for r in records if r.get("answer_mode") == "error")
    doc_hits = sum(1 for r in records if r.get("doc_id_hit"))
    section_hits = sum(1 for r in records if r.get("section_norm_hit"))
    zeros = sum(1 for r in records if r.get("citation_count", 0) == 0)
    routes: dict[str, int] = {}
    for r in records:
        route = r.get("route", "")
        routes[route] = routes.get(route, 0) + 1
    return {
        "sample_count": n,
        "empty_context_count": empty_ctx,
        "refusal_count": refusals,
        "empty_answer_count": empty_ans,
        "error_count": errors,
        "doc_id_hit_count": doc_hits,
        "section_norm_hit_count": section_hits,
        "zero_citation_count": zeros,
        "route_distribution": routes,
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    records = load_dataset(args.dataset)
    if args.max_samples > 0:
        records = records[:args.max_samples]
    print(f"[build_ragas_dataset] Loaded {len(records)} samples from {args.dataset}")

    # Fetch answers from RAG API
    include_debug = args.include_debug or (args.context_source == "final_chunks")
    if not include_debug and args.context_source == "cited_chunks":
        # cited_chunks mode still benefits from debug info
        include_debug = True

    enriched = fetch_answers(args.base_url, records, args.timeout, include_debug)

    # Convert to RAGAS format
    ragas_records = build_ragas_records(enriched, args.context_source, args.max_context_chars)

    # Write output
    output_path = output_dir / f"smoke100_ragas_dataset_{args.context_source}.jsonl"
    with output_path.open("w", encoding="utf-8") as fh:
        for rec in ragas_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write summary
    stats = summarize(ragas_records)
    summary_path = output_dir / f"smoke100_ragas_dataset_{args.context_source}_summary.json"
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[build_ragas_dataset] Wrote {len(ragas_records)} records → {output_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
