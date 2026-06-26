from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.synbio_rag.infrastructure.clients.openai_compatible import (  # noqa: E402
    OpenAICompatibleClient,
    extract_json_block,
)

DEFAULT_RESULTS_PATH = Path(
    "results/evaluation/"
    "v3_baseline_b0_parent_agg_b0_20260526_160749/"
    "b0_stable/results.jsonl"
)
DEFAULT_PARENT_CHUNKS_PATH = Path("data/paper_round1/chunks/parent_chunks.jsonl")
DEFAULT_CHILD_CHUNKS_PATH = Path("data/paper_round1/chunks/child_chunks.jsonl")
REPORTS_ROOT = Path("reports/evaluation")
MODEL_MAX_CANDIDATE_CHARS = 1200
CHILD_TEXT_MAX_CHARS = 350
PARENT_TEXT_MAX_CHARS = 500
BASELINE_SUPPORT_TEXT_MAX_CHARS = 500
PROMPT_VERSIONS = ("legacy", "id_only_v2")
SELECTION_MODES = ("replace", "append")


class SelectionValidationError(ValueError):
    pass


def main() -> None:
    args = parse_args()
    if args.target_only and args.controls_only:
        raise ValueError("--target-only and --controls-only cannot be used together")
    load_dotenv(".env")

    rows = load_jsonl(Path(args.results))
    parent_index = load_chunk_index(Path(args.parent_chunks))
    child_index = load_chunk_index(Path(args.child_chunks))
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir(selection_mode=args.selection_mode)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    env_config = resolve_deepseek_env(required=not args.dry_run)
    client = None
    if not args.dry_run:
        client = OpenAICompatibleClient(
            env_config["api_base"],
            env_config["api_key"],
            timeout_seconds=args.timeout_seconds,
        )

    result = run_replay(
        rows=rows,
        parent_index=parent_index,
        child_index=child_index,
        client=client,
        model_name=env_config.get("model_name", ""),
        output_dir=output_dir,
        dry_run=args.dry_run,
        include_targets=not args.controls_only,
        include_controls=not args.target_only,
        sample_ids=parse_sample_ids(args.sample_ids),
        limit=args.limit,
        max_tokens=args.max_tokens,
        selection_mode=args.selection_mode,
        prompt_version=args.prompt_version,
        max_extra_support=args.max_extra_support,
        direct_regression_source=Path(args.direct_regression_source)
        if args.direct_regression_source
        else None,
    )

    print(f"wrote report: {result['report_path']}")
    print(f"wrote summary: {result['summary_path']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay generation support selection with DeepSeek as an evidence selector."
    )
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--parent-chunks", default=str(DEFAULT_PARENT_CHUNKS_PATH))
    parser.add_argument("--child-chunks", default=str(DEFAULT_CHILD_CHUNKS_PATH))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--target-only", action="store_true")
    parser.add_argument("--controls-only", action="store_true")
    parser.add_argument("--selection-mode", choices=SELECTION_MODES, default="append")
    parser.add_argument("--prompt-version", choices=PROMPT_VERSIONS, default="id_only_v2")
    parser.add_argument("--max-extra-support", type=int, choices=(1, 2), default=1)
    parser.add_argument("--direct-regression-source", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=800)
    return parser.parse_args()


def default_output_dir(*, selection_mode: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPORTS_ROOT / f"generation_deepseek_support_selector_{selection_mode}_{run_id}"


def resolve_deepseek_env(*, required: bool) -> dict[str, str]:
    config = {
        "api_base": (os.getenv("DEEPSEEK_URL") or "").strip(),
        "api_key": (os.getenv("DEEPSEEK_API_KEY") or "").strip(),
        "model_name": (os.getenv("DEEPSEEK_MODEL_NAME") or "").strip(),
    }
    missing = [
        name
        for name, value in {
            "DEEPSEEK_URL": config["api_base"],
            "DEEPSEEK_API_KEY": config["api_key"],
            "DEEPSEEK_MODEL_NAME": config["model_name"],
        }.items()
        if not value
    ]
    if missing and required:
        raise RuntimeError(f"Missing DeepSeek env vars: {', '.join(missing)}")
    return config


def run_replay(
    *,
    rows: list[dict[str, Any]],
    parent_index: dict[str, dict[str, Any]],
    child_index: dict[str, dict[str, Any]],
    client: OpenAICompatibleClient | None,
    model_name: str,
    output_dir: Path,
    dry_run: bool,
    include_targets: bool,
    include_controls: bool,
    sample_ids: set[str],
    limit: int,
    max_tokens: int,
    selection_mode: str,
    prompt_version: str,
    max_extra_support: int,
    direct_regression_source: Path | None = None,
) -> dict[str, str]:
    prepared = prepare_samples(
        rows=rows,
        parent_index=parent_index,
        child_index=child_index,
        include_targets=include_targets,
        include_controls=include_controls,
        sample_ids=sample_ids,
        selection_mode=selection_mode,
        prompt_version=prompt_version,
        max_extra_support=max_extra_support,
    )
    callable_samples = [
        item
        for item in prepared
        if item["group"] in {"target_visible", "control"}
    ]
    callable_samples.sort(key=call_order_key)
    if limit > 0:
        callable_samples = callable_samples[:limit]

    selection_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    for sample in callable_samples:
        prompt_rows.append(
            {
                "sample_id": sample["sample_id"],
                "group": sample["group"],
                "selection_mode": sample["selection_mode"],
                "prompt_version": sample["prompt_version"],
                "max_select": sample["max_select"],
                "selectable_candidate_count": len(sample["selectable_candidates"]),
                "messages": sample["messages"],
            }
        )
        if dry_run:
            selection_rows.append(build_dry_run_selection(sample))
            continue
        if client is None:
            raise RuntimeError("client is required when dry_run is false")
        selection = call_selector_with_retry(
            client=client,
            model_name=model_name,
            sample=sample,
            max_tokens=max_tokens,
        )
        selection_rows.append(selection)
        if selection["status"] != "valid":
            invalid_rows.append(selection)

    skipped_rows = [
        build_skipped_selection(sample)
        for sample in prepared
        if sample["group"] in {"target_invisible", "out_of_scope"}
    ]
    all_rows = selection_rows + skipped_rows
    summary = summarize_replay(
        prepared=prepared,
        selection_rows=all_rows,
        dry_run=dry_run,
        include_controls=include_controls,
        include_targets=include_targets,
        limit=limit,
        model_name=model_name,
        selection_mode=selection_mode,
        prompt_version=prompt_version,
        max_extra_support=max_extra_support,
    )
    direct_review_rows = build_direct_regression_review_rows(
        source_path=resolve_direct_regression_source(
            direct_regression_source,
            output_dir=output_dir,
        ),
        limit=10,
    )
    if direct_review_rows:
        summary["direct_replacement_regression_review"] = {
            "source_path": direct_review_rows[0]["source_path"],
            "total_regressed_count": direct_review_rows[0]["total_regressed_count"],
            "sample_count": len(direct_review_rows),
            "sample_ids": [row["sample_id"] for row in direct_review_rows],
            "manual_label_options": direct_review_rows[0]["manual_label_options"],
        }

    write_jsonl(output_dir / "model_selections.jsonl", all_rows)
    write_json(output_dir / "summary.json", summary)
    write_markdown(output_dir / "report.md", render_report(summary, all_rows))
    if direct_review_rows:
        write_jsonl(
            output_dir / "direct_replacement_regression_review_sample.jsonl",
            direct_review_rows,
        )
    if invalid_rows:
        write_jsonl(output_dir / "invalid_outputs.jsonl", invalid_rows)
    if dry_run:
        write_jsonl(output_dir / "dry_run_prompts.jsonl", prompt_rows)

    return {
        "report_path": str(output_dir / "report.md"),
        "summary_path": str(output_dir / "summary.json"),
    }


def prepare_samples(
    *,
    rows: list[dict[str, Any]],
    parent_index: dict[str, dict[str, Any]],
    child_index: dict[str, dict[str, Any]],
    include_targets: bool,
    include_controls: bool,
    sample_ids: set[str],
    selection_mode: str = "append",
    prompt_version: str = "id_only_v2",
    max_extra_support: int = 1,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if sample_ids and sample_id not in sample_ids:
            continue
        group = classify_sample(row)
        if group == "control" and not include_controls:
            continue
        if group in {"target_visible", "target_invisible"} and not include_targets:
            continue
        if group == "out_of_scope" and not sample_ids:
            continue
        candidates = build_model_candidates(
            row=row,
            parent_index=parent_index,
            child_index=child_index,
        )
        baseline_support = build_baseline_support(
            row=row,
            candidates=candidates,
            parent_index=parent_index,
            child_index=child_index,
        )
        selectable_candidates = selectable_candidates_for_mode(
            selection_mode=selection_mode,
            candidates=candidates,
            baseline_support=baseline_support,
        )
        max_select = max_select_for_mode(
            selection_mode=selection_mode,
            row=row,
            candidate_count=len(selectable_candidates),
            max_extra_support=max_extra_support,
        )
        payload = build_prompt_payload(
            row,
            selectable_candidates,
            baseline_support=baseline_support,
            max_select=max_select,
            selection_mode=selection_mode,
            prompt_version=prompt_version,
        )
        prepared.append(
            {
                "sample_id": sample_id,
                "group": group,
                "row": row,
                "candidates": candidates,
                "baseline_support": baseline_support,
                "selectable_candidates": selectable_candidates,
                "max_select": max_select,
                "selection_mode": selection_mode,
                "prompt_version": prompt_version,
                "allow_empty_selection": selection_mode == "append",
                "messages": build_selector_messages(payload),
                "baseline": baseline_metrics(row),
            }
        )
    return prepared


def call_order_key(sample: dict[str, Any]) -> tuple[int, str]:
    priority = 0 if sample["group"] == "target_visible" else 1
    return priority, str(sample["sample_id"])


def classify_sample(row: dict[str, Any]) -> str:
    metrics = row.get("rule_metrics") or {}
    parent_hit = metrics.get("parent_chunk_hit_at10") is True
    support_hit = metrics.get("support_parent_chunk_hit")
    if parent_hit and support_hit is False:
        return "target_visible" if gold_parent_in_candidates(row) else "target_invisible"
    if parent_hit and support_hit is True:
        return "control"
    return "out_of_scope"


def gold_parent_in_candidates(row: dict[str, Any]) -> bool:
    gold = set(str(value) for value in row.get("gold_parent_chunk_ids") or [])
    candidates = generation_debug(row).get("candidates") or []
    return any(parent_id_from_item(item) in gold for item in candidates if isinstance(item, dict))


def build_model_candidates(
    *,
    row: dict[str, Any],
    parent_index: dict[str, dict[str, Any]],
    child_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = generation_debug(row).get("candidates") or []
    model_candidates: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        candidate = {
            "evidence_id": str(item.get("evidence_id") or ""),
            "chunk_id": str(item.get("chunk_id") or ""),
            "parent_chunk_id": parent_id_from_item(item),
            "doc_id": str(item.get("doc_id") or ""),
            "section": str(item.get("section") or ""),
            "rerank_rank": item.get("rerank_rank"),
            "scores": item.get("scores") or {},
            "features": item.get("features") or {},
            "matched_child_chunk_ids": [
                str(value) for value in item.get("matched_child_chunk_ids") or []
            ],
            "generation_evidence_role": str(item.get("generation_evidence_role") or ""),
            "evidence_text": build_candidate_text(
                item,
                parent_index=parent_index,
                child_index=child_index,
            ),
        }
        if candidate["evidence_id"]:
            model_candidates.append(candidate)
    return model_candidates


def build_candidate_text(
    item: dict[str, Any],
    *,
    parent_index: dict[str, dict[str, Any]],
    child_index: dict[str, dict[str, Any]],
) -> str:
    parts: list[str] = []
    for child_id in [str(value) for value in item.get("matched_child_chunk_ids") or []][:4]:
        child = child_index.get(child_id)
        if not child:
            continue
        text = clean_text(str(child.get("text") or child.get("retrieval_text") or ""))
        if text:
            parts.append(f"matched child {child_id}: {clip_text(text, CHILD_TEXT_MAX_CHARS)}")

    parent_id = parent_id_from_item(item)
    parent = parent_index.get(parent_id)
    if parent:
        text = clean_text(str(parent.get("text") or parent.get("retrieval_text") or ""))
        if text:
            parts.append(f"parent excerpt {parent_id}: {clip_text(text, PARENT_TEXT_MAX_CHARS)}")

    if not parts:
        parts.append(str(item.get("text_preview") or ""))
    return clip_text("\n".join(parts), MODEL_MAX_CANDIDATE_CHARS)


def build_baseline_support(
    *,
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    parent_index: dict[str, dict[str, Any]],
    child_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_by_evidence_id = {
        candidate["evidence_id"]: candidate
        for candidate in candidates
        if candidate.get("evidence_id")
    }
    support_pack = generation_debug(row).get("support_pack") or []
    baseline: list[dict[str, Any]] = []
    seen = set()
    if isinstance(support_pack, list):
        for item in support_pack:
            if not isinstance(item, dict):
                continue
            evidence_id = str(item.get("evidence_id") or "")
            source = candidate_by_evidence_id.get(evidence_id)
            support_item = dict(source) if source else build_support_item_from_debug(
                item,
                parent_index=parent_index,
                child_index=child_index,
            )
            key = support_identity_key(support_item)
            if key and key not in seen:
                baseline.append(support_item)
                seen.add(key)

    baseline_chunk_ids = {str(value) for value in row.get("support_chunk_ids") or []}
    if not baseline_chunk_ids:
        return baseline
    for candidate in candidates:
        candidate_chunk_ids = {
            str(candidate.get("chunk_id") or ""),
            str(candidate.get("parent_chunk_id") or ""),
        }
        if not (baseline_chunk_ids & candidate_chunk_ids):
            continue
        key = support_identity_key(candidate)
        if key and key not in seen:
            baseline.append(dict(candidate))
            seen.add(key)
    return baseline


def build_support_item_from_debug(
    item: dict[str, Any],
    *,
    parent_index: dict[str, dict[str, Any]],
    child_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "evidence_id": str(item.get("evidence_id") or ""),
        "chunk_id": str(item.get("chunk_id") or ""),
        "parent_chunk_id": parent_id_from_item(item),
        "doc_id": str(item.get("doc_id") or ""),
        "section": str(item.get("section") or ""),
        "rerank_rank": item.get("rerank_rank"),
        "scores": item.get("scores") or {},
        "features": item.get("features") or {},
        "matched_child_chunk_ids": [
            str(value) for value in item.get("matched_child_chunk_ids") or []
        ],
        "generation_evidence_role": str(item.get("generation_evidence_role") or ""),
        "evidence_text": build_candidate_text(
            item,
            parent_index=parent_index,
            child_index=child_index,
        ),
    }


def selectable_candidates_for_mode(
    *,
    selection_mode: str,
    candidates: list[dict[str, Any]],
    baseline_support: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if selection_mode == "replace":
        return list(candidates)

    baseline_evidence_ids = {
        str(item.get("evidence_id") or "") for item in baseline_support if item.get("evidence_id")
    }
    baseline_chunk_ids = {
        str(value)
        for item in baseline_support
        for value in (item.get("chunk_id"), item.get("parent_chunk_id"))
        if value
    }
    return [
        candidate
        for candidate in candidates
        if str(candidate.get("evidence_id") or "") not in baseline_evidence_ids
        and str(candidate.get("chunk_id") or "") not in baseline_chunk_ids
        and str(candidate.get("parent_chunk_id") or "") not in baseline_chunk_ids
    ]


def max_select_for_mode(
    *,
    selection_mode: str,
    row: dict[str, Any],
    candidate_count: int,
    max_extra_support: int,
) -> int:
    if selection_mode == "append":
        return max(0, min(max_extra_support, candidate_count))
    return support_pack_size(row, candidate_count=candidate_count)


def support_identity_key(item: dict[str, Any]) -> tuple[str, str] | None:
    evidence_id = str(item.get("evidence_id") or "")
    if evidence_id:
        return ("evidence_id", evidence_id)
    parent_id = str(item.get("parent_chunk_id") or item.get("chunk_id") or "")
    return ("parent_chunk_id", parent_id) if parent_id else None


def build_prompt_payload(
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    baseline_support: list[dict[str, Any]] | None = None,
    max_select: int,
    selection_mode: str = "append",
    prompt_version: str = "id_only_v2",
) -> dict[str, Any]:
    if prompt_version == "id_only_v2":
        return build_id_only_v2_prompt_payload(
            row=row,
            candidates=candidates,
            baseline_support=baseline_support or [],
            max_select=max_select,
            selection_mode=selection_mode,
        )

    return {
        "task": "select_support_evidence_ids",
        "sample_id": row.get("sample_id"),
        "question": row.get("question"),
        "actual_route": row.get("actual_route"),
        "category": row.get("category"),
        "max_select": max_select,
        "selection_rules": [
            "Only choose evidence_id values from candidates.",
            "Choose evidence that directly answers the question.",
            "Prefer concrete table, figure, numeric, result, or method evidence when relevant.",
            "Return JSON only.",
        ],
        "output_schema": {
            "selected_evidence_ids": ["E1"],
            "reason": "short reason",
            "confidence": 0.0,
        },
        "candidates": candidates,
    }


def build_id_only_v2_prompt_payload(
    *,
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    baseline_support: list[dict[str, Any]],
    max_select: int,
    selection_mode: str,
) -> dict[str, Any]:
    if selection_mode == "append":
        return {
            "task": "append_extra_support_evidence_ids",
            "sample_id": row.get("sample_id"),
            "question": row.get("question"),
            "actual_route": row.get("actual_route"),
            "max_extra_support": max_select,
            "selection_rules": [
                "Baseline support is already kept and must not be re-selected.",
                "Only choose evidence_id values from append_candidates.",
                "Choose at most max_extra_support additional evidence items.",
                "If no append candidate materially improves support, return {\"selected_evidence_ids\":[]}.",
                "Return exactly one JSON object and never return empty content.",
            ],
            "output_schema": {"selected_evidence_ids": ["E5"]},
            "baseline_support_summary": [
                prompt_support_summary(item) for item in baseline_support
            ],
            "append_candidates": [prompt_candidate_summary(item) for item in candidates],
        }

    return {
        "task": "select_support_evidence_ids",
        "sample_id": row.get("sample_id"),
        "question": row.get("question"),
        "actual_route": row.get("actual_route"),
        "max_select": max_select,
        "selection_rules": [
            "Only choose evidence_id values from candidates.",
            "Choose evidence that directly answers the question.",
            "Return exactly one JSON object and never return empty content.",
        ],
        "output_schema": {"selected_evidence_ids": ["E1"]},
        "candidates": [prompt_candidate_summary(item) for item in candidates],
    }


def prompt_candidate_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": item.get("evidence_id"),
        "parent_chunk_id": item.get("parent_chunk_id"),
        "doc_id": item.get("doc_id"),
        "section": item.get("section"),
        "rerank_rank": item.get("rerank_rank"),
        "generation_evidence_role": item.get("generation_evidence_role"),
        "matched_child_chunk_ids": item.get("matched_child_chunk_ids") or [],
        "evidence_text": item.get("evidence_text"),
    }


def prompt_support_summary(item: dict[str, Any]) -> dict[str, Any]:
    summary = prompt_candidate_summary(item)
    summary["evidence_text"] = clip_text(
        str(summary.get("evidence_text") or ""),
        BASELINE_SUPPORT_TEXT_MAX_CHARS,
    )
    return summary


def build_selector_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是 RAG 证据选择器。你只选择已给出的 evidence_id，不能编造 ID。"
                "只输出一个合法 JSON 对象，不要输出 Markdown。"
                '如果不选择证据，也必须输出 {"selected_evidence_ids":[]}，不能输出空内容。'
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]


def call_selector_with_retry(
    *,
    client: OpenAICompatibleClient,
    model_name: str,
    sample: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    allowed_ids = {
        candidate["evidence_id"] for candidate in sample["selectable_candidates"]
    }
    if not allowed_ids and sample.get("allow_empty_selection"):
        return build_valid_selection(
            sample,
            {"selected_evidence_ids": [], "reason": "", "confidence": 0.0},
            raw_output="",
        )
    messages = list(sample["messages"])
    raw_outputs: list[str] = []
    errors: list[str] = []
    for attempt in range(2):
        try:
            raw = client.chat_completion(
                model=model_name,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
            )
            raw_outputs.append(raw)
            parsed = extract_json_block(raw)
            selection = validate_model_selection(
                parsed,
                allowed_ids=allowed_ids,
                max_select=sample["max_select"],
                allow_empty=sample.get("allow_empty_selection", False),
            )
            return build_valid_selection(sample, selection, raw_output=raw)
        except Exception as exc:  # noqa: BLE001 - preserve model/debug failure as data
            errors.append(f"{type(exc).__name__}: {exc}")
            messages = repair_messages(
                sample["messages"],
                errors[-1],
                allow_empty=sample.get("allow_empty_selection", False),
            )
            if attempt == 1:
                break
    return build_invalid_selection(sample, errors=errors, raw_outputs=raw_outputs)


def validate_model_selection(
    value: Any,
    *,
    allowed_ids: set[str],
    max_select: int,
    allow_empty: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SelectionValidationError("model output must be a JSON object")
    raw_ids = value.get("selected_evidence_ids")
    if not isinstance(raw_ids, list):
        raise SelectionValidationError("selected_evidence_ids must be a list")
    selected_ids: list[str] = []
    for raw_id in raw_ids:
        evidence_id = str(raw_id or "").strip()
        if not evidence_id:
            continue
        if evidence_id not in allowed_ids:
            raise SelectionValidationError(f"unknown evidence_id: {evidence_id}")
        if evidence_id not in selected_ids:
            selected_ids.append(evidence_id)
    if not selected_ids and not allow_empty:
        raise SelectionValidationError("selected_evidence_ids must not be empty")
    if len(selected_ids) > max_select:
        raise SelectionValidationError(
            f"selected_evidence_ids exceeds max_select={max_select}"
        )
    return {
        "selected_evidence_ids": selected_ids,
        "reason": str(value.get("reason") or ""),
        "confidence": parse_float(value.get("confidence")),
    }


def repair_messages(
    messages: list[dict[str, str]],
    error: str,
    *,
    allow_empty: bool = False,
) -> list[dict[str, str]]:
    empty_note = "可以返回空数组。" if allow_empty else "至少选择 1 条。"
    repaired = list(messages)
    repaired.append(
        {
            "role": "user",
            "content": (
                "上一次输出无效："
                f"{error}\n"
                "请重新输出合法 JSON，格式为 "
                '{"selected_evidence_ids":["E1"]}。'
                f"{empty_note}"
                '如果不选择证据，输出 {"selected_evidence_ids":[]}。'
                "只能选择候选中的 evidence_id，不能输出空内容。"
            ),
        }
    )
    return repaired


def build_valid_selection(
    sample: dict[str, Any],
    selection: dict[str, Any],
    *,
    raw_output: str,
) -> dict[str, Any]:
    if sample["selection_mode"] == "append":
        metrics = compute_append_metrics(
            row=sample["row"],
            candidates=sample["candidates"],
            baseline_support=sample["baseline_support"],
            selected_extra_evidence_ids=selection["selected_evidence_ids"],
        )
        outcome = classify_append_outcome(sample["group"], metrics)
    else:
        metrics = compute_model_metrics(
            row=sample["row"],
            candidates=sample["candidates"],
            selected_evidence_ids=selection["selected_evidence_ids"],
        )
        outcome = classify_outcome(sample["group"], metrics)
    return {
        "sample_id": sample["sample_id"],
        "group": sample["group"],
        "selection_mode": sample["selection_mode"],
        "prompt_version": sample["prompt_version"],
        "status": "valid",
        "selected_evidence_ids": selection["selected_evidence_ids"],
        "reason": selection["reason"],
        "confidence": selection["confidence"],
        "raw_output_preview": raw_output[:500],
        "baseline": sample["baseline"],
        "model_metrics": metrics,
        "outcome": outcome,
    }


def build_invalid_selection(
    sample: dict[str, Any],
    *,
    errors: list[str],
    raw_outputs: list[str],
) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "group": sample["group"],
        "selection_mode": sample["selection_mode"],
        "prompt_version": sample["prompt_version"],
        "status": "invalid",
        "selected_evidence_ids": [],
        "errors": errors,
        "raw_output_preview": raw_outputs[-1][:500] if raw_outputs else "",
        "baseline": sample["baseline"],
        "model_metrics": {},
        "outcome": "model_invalid_output",
    }


def build_dry_run_selection(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "group": sample["group"],
        "selection_mode": sample["selection_mode"],
        "prompt_version": sample["prompt_version"],
        "status": "dry_run",
        "selected_evidence_ids": [],
        "baseline": sample["baseline"],
        "model_metrics": {},
        "outcome": "dry_run_not_called",
    }


def build_skipped_selection(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "group": sample["group"],
        "selection_mode": sample["selection_mode"],
        "prompt_version": sample["prompt_version"],
        "status": "skipped",
        "selected_evidence_ids": [],
        "baseline": sample["baseline"],
        "model_metrics": {},
        "outcome": "model_cannot_see_gold_parent"
        if sample["group"] == "target_invisible"
        else "out_of_scope",
    }


def compute_model_metrics(
    *,
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    selected_evidence_ids: list[str],
) -> dict[str, Any]:
    selected = [
        candidate
        for candidate in candidates
        if candidate.get("evidence_id") in set(selected_evidence_ids)
    ]
    support = collect_support_fields(selected)
    hits = compute_support_hits(row=row, selected=selected)
    return {
        "selected_parent_chunk_ids": support["parent_chunk_ids"],
        "selected_matched_child_chunk_ids": support["matched_child_chunk_ids"],
        "model_support_parent_hit": hits["support_parent_hit"],
        "model_support_child_evidence_hit": hits["support_child_evidence_hit"],
    }


def compute_append_metrics(
    *,
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    baseline_support: list[dict[str, Any]],
    selected_extra_evidence_ids: list[str],
) -> dict[str, Any]:
    selected_id_set = set(selected_extra_evidence_ids)
    extra_support = [
        candidate
        for candidate in candidates
        if candidate.get("evidence_id") in selected_id_set
    ]
    hybrid_support = dedupe_support_items([*baseline_support, *extra_support])

    baseline_fields = collect_support_fields(baseline_support)
    extra_fields = collect_support_fields(extra_support)
    hybrid_fields = collect_support_fields(hybrid_support)
    baseline_hits = compute_support_hits(row=row, selected=baseline_support)
    hybrid_hits = compute_support_hits(row=row, selected=hybrid_support)
    gold_parents = set(str(value) for value in row.get("gold_parent_chunk_ids") or [])
    extra_noise_count = sum(
        1
        for item in extra_support
        if gold_parents
        and str(item.get("parent_chunk_id") or "") not in gold_parents
    )
    rescued_by_append = (
        baseline_hits["support_parent_hit"] is not True
        and hybrid_hits["support_parent_hit"] is True
    )

    return {
        "baseline_support_evidence_ids": baseline_fields["evidence_ids"],
        "baseline_support_parent_chunk_ids": baseline_fields["parent_chunk_ids"],
        "baseline_support_matched_child_chunk_ids": baseline_fields[
            "matched_child_chunk_ids"
        ],
        "baseline_support_parent_hit_recomputed": baseline_hits["support_parent_hit"],
        "baseline_support_child_evidence_hit_recomputed": baseline_hits[
            "support_child_evidence_hit"
        ],
        "extra_support_evidence_ids": extra_fields["evidence_ids"],
        "extra_support_parent_chunk_ids": extra_fields["parent_chunk_ids"],
        "extra_support_matched_child_chunk_ids": extra_fields["matched_child_chunk_ids"],
        "hybrid_support_evidence_ids": hybrid_fields["evidence_ids"],
        "hybrid_support_parent_chunk_ids": hybrid_fields["parent_chunk_ids"],
        "hybrid_support_matched_child_chunk_ids": hybrid_fields["matched_child_chunk_ids"],
        "hybrid_support_parent_hit": hybrid_hits["support_parent_hit"],
        "hybrid_support_child_evidence_hit": hybrid_hits["support_child_evidence_hit"],
        "model_support_parent_hit": hybrid_hits["support_parent_hit"],
        "model_support_child_evidence_hit": hybrid_hits["support_child_evidence_hit"],
        "rescued_by_append": rescued_by_append,
        "extra_support_count": len(extra_support),
        "extra_noise_count": extra_noise_count,
        "extra_rerank_ranks": [
            item.get("rerank_rank") for item in extra_support if item.get("rerank_rank") is not None
        ],
    }


def collect_support_fields(items: list[dict[str, Any]]) -> dict[str, list[str]]:
    evidence_ids: list[str] = []
    parent_ids: set[str] = set()
    child_ids: set[str] = set()
    for item in items:
        evidence_id = str(item.get("evidence_id") or "")
        if evidence_id and evidence_id not in evidence_ids:
            evidence_ids.append(evidence_id)
        parent_id = str(item.get("parent_chunk_id") or "")
        if parent_id:
            parent_ids.add(parent_id)
        child_ids.update(str(value) for value in item.get("matched_child_chunk_ids") or [])
    return {
        "evidence_ids": evidence_ids,
        "parent_chunk_ids": sorted(parent_ids),
        "matched_child_chunk_ids": sorted(child_ids),
    }


def compute_support_hits(
    *,
    row: dict[str, Any],
    selected: list[dict[str, Any]],
) -> dict[str, bool | None]:
    gold_parents = set(str(value) for value in row.get("gold_parent_chunk_ids") or [])
    gold_children = {
        str(value)
        for value in row.get("gold_chunk_ids") or []
        if "::child" in str(value)
    }
    selected_parents = {str(candidate.get("parent_chunk_id") or "") for candidate in selected}
    selected_children: set[str] = set()
    for candidate in selected:
        selected_children.update(
            str(value) for value in candidate.get("matched_child_chunk_ids") or []
        )
    return {
        "support_parent_hit": bool(gold_parents & selected_parents)
        if gold_parents
        else None,
        "support_child_evidence_hit": bool(gold_children & selected_children)
        if gold_children
        else None,
    }


def dedupe_support_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen = set()
    for item in items:
        key = support_identity_key(item)
        if not key or key in seen:
            continue
        deduped.append(item)
        seen.add(key)
    return deduped


def classify_outcome(group: str, model_metrics: dict[str, Any]) -> str:
    parent_hit = model_metrics.get("model_support_parent_hit") is True
    if group == "target_visible":
        return "rescued" if parent_hit else "still_miss"
    if group == "control":
        return "retained" if parent_hit else "regressed"
    return "not_scored"


def classify_append_outcome(group: str, model_metrics: dict[str, Any]) -> str:
    parent_hit = model_metrics.get("hybrid_support_parent_hit") is True
    if group == "target_visible":
        return "rescued_by_append" if model_metrics.get("rescued_by_append") else "still_miss"
    if group == "control":
        return "retained_by_append" if parent_hit else "unexpected_append_regression"
    return "not_scored"


def summarize_replay(
    *,
    prepared: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    dry_run: bool,
    include_targets: bool,
    include_controls: bool,
    limit: int,
    model_name: str,
    selection_mode: str,
    prompt_version: str,
    max_extra_support: int,
) -> dict[str, Any]:
    by_group = Counter(item["group"] for item in prepared)
    status_counts = Counter(row["status"] for row in selection_rows)
    outcome_counts = Counter(row["outcome"] for row in selection_rows)
    status_counts_by_group: dict[str, dict[str, int]] = {}
    for group in sorted({row["group"] for row in selection_rows}):
        status_counts_by_group[group] = dict(
            Counter(row["status"] for row in selection_rows if row["group"] == group)
        )
    target_visible_ids = [
        item["sample_id"] for item in prepared if item["group"] == "target_visible"
    ]
    target_invisible_ids = [
        item["sample_id"] for item in prepared if item["group"] == "target_invisible"
    ]
    control_ids = [item["sample_id"] for item in prepared if item["group"] == "control"]
    valid_rows = [row for row in selection_rows if row["status"] == "valid"]
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "model_name": model_name if model_name else "not_configured",
        "selection_mode": selection_mode,
        "prompt_version": prompt_version,
        "max_extra_support": max_extra_support,
        "include_targets": include_targets,
        "include_controls": include_controls,
        "limit": limit,
        "sample_counts": dict(by_group),
        "status_counts": dict(status_counts),
        "status_counts_by_group": status_counts_by_group,
        "outcome_counts": dict(outcome_counts),
        "target_visible_sample_ids": target_visible_ids,
        "target_invisible_sample_ids": target_invisible_ids,
        "control_sample_count": len(control_ids),
        "called_sample_count": sum(
            1 for row in selection_rows if row["status"] in {"valid", "invalid", "dry_run"}
        ),
        "valid_sample_count": len(valid_rows),
        "by_category": summarize_by_field(valid_rows, field="category"),
        "by_expected_route": summarize_by_field(valid_rows, field="expected_route"),
        "by_route_match": summarize_by_field(valid_rows, field="route_match"),
        "append_metrics": summarize_append_metrics(valid_rows),
    }


def summarize_append_metrics(selection_rows: list[dict[str, Any]]) -> dict[str, Any]:
    append_rows = [
        row
        for row in selection_rows
        if row.get("selection_mode") == "append" and row.get("status") == "valid"
    ]
    if not append_rows:
        return {}

    target_rescued_ids = [
        row["sample_id"]
        for row in append_rows
        if (row.get("model_metrics") or {}).get("rescued_by_append") is True
    ]
    control_regression_ids = [
        row["sample_id"]
        for row in append_rows
        if row.get("group") == "control"
        and (row.get("model_metrics") or {}).get("hybrid_support_parent_hit") is not True
    ]
    extra_counts = [
        int((row.get("model_metrics") or {}).get("extra_support_count") or 0)
        for row in append_rows
    ]
    extra_noise_counts = [
        int((row.get("model_metrics") or {}).get("extra_noise_count") or 0)
        for row in append_rows
    ]
    evidence_id_distribution: Counter[str] = Counter()
    parent_distribution: Counter[str] = Counter()
    rerank_distribution: Counter[str] = Counter()
    for row in append_rows:
        metrics = row.get("model_metrics") or {}
        evidence_id_distribution.update(metrics.get("extra_support_evidence_ids") or [])
        parent_distribution.update(metrics.get("extra_support_parent_chunk_ids") or [])
        rerank_distribution.update(str(value) for value in metrics.get("extra_rerank_ranks") or [])

    return {
        "valid_append_sample_count": len(append_rows),
        "rescued_by_append_count": len(target_rescued_ids),
        "rescued_by_append_sample_ids": target_rescued_ids,
        "control_exact_hit_regression_count": len(control_regression_ids),
        "control_exact_hit_regression_sample_ids": control_regression_ids,
        "extra_support_count_total": sum(extra_counts),
        "extra_support_count_average": round(sum(extra_counts) / len(extra_counts), 3)
        if extra_counts
        else 0.0,
        "extra_support_count_distribution": stringify_counter(Counter(extra_counts)),
        "extra_noise_count_total": sum(extra_noise_counts),
        "extra_noise_count_average": round(sum(extra_noise_counts) / len(extra_noise_counts), 3)
        if extra_noise_counts
        else 0.0,
        "extra_noise_count_distribution": stringify_counter(Counter(extra_noise_counts)),
        "extra_evidence_id_distribution": dict(evidence_id_distribution),
        "extra_rerank_rank_distribution": dict(rerank_distribution),
        "extra_parent_chunk_id_top20": dict(parent_distribution.most_common(20)),
    }


def stringify_counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(counter.items())}


def resolve_direct_regression_source(
    explicit_source: Path | None,
    *,
    output_dir: Path,
) -> Path | None:
    if explicit_source:
        if not explicit_source.exists():
            raise FileNotFoundError(f"direct regression source not found: {explicit_source}")
        return explicit_source

    candidates = sorted(
        REPORTS_ROOT.glob(
            "generation_deepseek_support_selector_controls_*/model_selections.jsonl"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if not is_relative_to(candidate.resolve(), output_dir.resolve()):
            return candidate
    return None


def build_direct_regression_review_rows(
    *,
    source_path: Path | None,
    limit: int,
) -> list[dict[str, Any]]:
    if source_path is None:
        return []
    rows = load_jsonl(source_path)
    regressed = [row for row in rows if row.get("outcome") == "regressed"]
    label_options = ["模型错选", "可接受替代证据", "gold 粒度过细", "prompt 误导"]
    review_rows: list[dict[str, Any]] = []
    for row in regressed[:limit]:
        review_rows.append(
            {
                "source_path": str(source_path),
                "total_regressed_count": len(regressed),
                "sample_id": row.get("sample_id"),
                "previous_selected_evidence_ids": row.get("selected_evidence_ids") or [],
                "previous_reason": row.get("reason") or "",
                "previous_model_metrics": row.get("model_metrics") or {},
                "baseline": row.get("baseline") or {},
                "manual_label": "待人工复核",
                "manual_label_options": label_options,
            }
        )
    return review_rows


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def summarize_by_field(selection_rows: list[dict[str, Any]], *, field: str) -> dict[str, Any]:
    buckets: dict[str, Counter] = defaultdict(Counter)
    for row in selection_rows:
        baseline = row.get("baseline") or {}
        value = str(baseline.get(field))
        buckets[value][row.get("outcome", "")] += 1
    return {key: dict(counter) for key, counter in sorted(buckets.items())}


def baseline_metrics(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("rule_metrics") or {}
    return {
        "category": row.get("category"),
        "expected_route": row.get("expected_route"),
        "actual_route": row.get("actual_route"),
        "route_match": metrics.get("route_match"),
        "parent_chunk_hit_at10": metrics.get("parent_chunk_hit_at10"),
        "support_parent_chunk_hit": metrics.get("support_parent_chunk_hit"),
        "support_child_evidence_hit": metrics.get("support_child_evidence_hit"),
        "citation_parent_chunk_hit": metrics.get("citation_parent_chunk_hit"),
    }


def support_pack_size(row: dict[str, Any], *, candidate_count: int) -> int:
    support_pack = generation_debug(row).get("support_pack") or []
    size = len(support_pack) if isinstance(support_pack, list) else 0
    if size <= 0:
        size = len(row.get("support_chunk_ids") or [])
    if size <= 0:
        size = 3
    return max(1, min(size, candidate_count or 1))


def render_report(summary: dict[str, Any], selection_rows: list[dict[str, Any]]) -> str:
    append_metrics = summary.get("append_metrics") or {}
    lines = [
        "# DeepSeek Support Selector 离线 Replay",
        "",
        "## 结论摘要",
        "",
    ]
    if summary["dry_run"]:
        lines.append("本次是 dry-run，只构造 prompt 和样本分组，没有调用 DeepSeek。")
    else:
        lines.append("本次调用 DeepSeek 只选择 evidence_id，没有生成答案，也没有调用 Judge。")
    lines.extend(
        [
            "",
            f"- 模型：`{summary['model_name']}`",
            f"- selection mode：`{summary['selection_mode']}`",
            f"- prompt version：`{summary['prompt_version']}`",
            f"- max extra support：`{summary['max_extra_support']}`",
            f"- target visible：`{summary['sample_counts'].get('target_visible', 0)}`",
            f"- target invisible：`{summary['sample_counts'].get('target_invisible', 0)}`",
            f"- controls：`{summary['sample_counts'].get('control', 0)}`",
            f"- valid calls：`{summary['valid_sample_count']}`",
            "",
            "## Outcome",
            "",
            "| outcome | count |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(summary["outcome_counts"].items()):
        lines.append(f"| `{key}` | {value} |")

    if append_metrics:
        lines.extend(
            [
                "",
                "## Append 指标",
                "",
                "| metric | value |",
                "|---|---:|",
                f"| rescued_by_append | {append_metrics['rescued_by_append_count']} |",
                f"| controls exact-hit regression | {append_metrics['control_exact_hit_regression_count']} |",
                f"| extra support total | {append_metrics['extra_support_count_total']} |",
                f"| extra support avg | {append_metrics['extra_support_count_average']} |",
                f"| extra noise total | {append_metrics['extra_noise_count_total']} |",
                f"| extra noise avg | {append_metrics['extra_noise_count_average']} |",
                "",
                f"- rescued sample ids：{format_ids(append_metrics['rescued_by_append_sample_ids'])}",
                f"- control regression ids：{format_ids(append_metrics['control_exact_hit_regression_sample_ids'])}",
                f"- extra support count distribution：`{append_metrics['extra_support_count_distribution']}`",
                f"- extra noise count distribution：`{append_metrics['extra_noise_count_distribution']}`",
                f"- extra evidence id distribution：`{append_metrics['extra_evidence_id_distribution']}`",
                f"- extra rerank rank distribution：`{append_metrics['extra_rerank_rank_distribution']}`",
            ]
        )
    direct_review = summary.get("direct_replacement_regression_review") or {}
    if direct_review:
        lines.extend(
            [
                "",
                "## Direct Replacement 回退复核",
                "",
                f"- source：`{direct_review['source_path']}`",
                f"- regressed total：`{direct_review['total_regressed_count']}`",
                f"- sampled for manual review：{format_ids(direct_review['sample_ids'])}",
                "- manual label options："
                + ", ".join(f"`{value}`" for value in direct_review["manual_label_options"]),
                "",
                "详细清单写入 `direct_replacement_regression_review_sample.jsonl`，"
                "默认标签为 `待人工复核`。",
            ]
        )
    lines.extend(
        [
            "",
            "## Target 样本",
            "",
            "模型可见的 target 是 gold parent 已进入 final candidates 的 support miss；"
            "模型不可见的 target 是 gold parent 只在 retrieved top10，未进入 final candidates。",
            "",
            f"- visible：{format_ids(summary['target_visible_sample_ids'])}",
            f"- invisible：{format_ids(summary['target_invisible_sample_ids'])}",
            "",
            "target_invisible 样本的 gold parent 未进入 final candidates，模型在本 replay 中不可见；"
            "这些样本继续单独列出，不计为模型选择失败。",
            "",
            "## 按 Category",
            "",
            render_nested_counter(summary["by_category"]),
            "",
            "## 按 Expected Route",
            "",
            render_nested_counter(summary["by_expected_route"]),
            "",
            "## 样本结果",
            "",
            "| sample_id | group | status | outcome | selected_evidence_ids | hybrid_parent_hit | extra_noise_count |",
            "|---|---|---|---|---|---|---:|",
        ]
    )
    for row in selection_rows:
        metrics = row.get("model_metrics") or {}
        hybrid_parent_hit = metrics.get("hybrid_support_parent_hit")
        if hybrid_parent_hit is None:
            hybrid_parent_hit = metrics.get("model_support_parent_hit", "")
        lines.append(
            "| "
            f"`{row['sample_id']}` | `{row['group']}` | `{row['status']}` | "
            f"`{row['outcome']}` | `{', '.join(row.get('selected_evidence_ids') or [])}` |"
            f" `{hybrid_parent_hit}` | {metrics.get('extra_noise_count', '')} |"
        )
    return "\n".join(lines)


def render_nested_counter(value: dict[str, Any]) -> str:
    if not value:
        return "无有效模型结果。"
    lines = ["| bucket | outcomes |", "|---|---|"]
    for bucket, counts in value.items():
        parts = ", ".join(f"{key}={count}" for key, count in sorted(counts.items()))
        lines.append(f"| `{bucket}` | {parts} |")
    return "\n".join(lines)


def format_ids(values: list[str]) -> str:
    if not values:
        return "-"
    return ", ".join(f"`{value}`" for value in values)


def generation_debug(row: dict[str, Any]) -> dict[str, Any]:
    debug = row.get("debug_digest") or {}
    generation = debug.get("generation_v2") or {}
    return generation if isinstance(generation, dict) else {}


def parent_id_from_item(item: dict[str, Any]) -> str:
    value = item.get("parent_chunk_id")
    if value:
        return str(value)
    return parent_chunk_id(item.get("chunk_id"))


def parent_chunk_id(chunk_id: Any) -> str:
    return str(chunk_id or "").split("::child", 1)[0]


def parse_sample_ids(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def clean_text(text: str) -> str:
    return " ".join((text or "").split())


def clip_text(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def safe_url(url: str) -> str:
    if not url:
        return ""
    return url.split("?", 1)[0]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_chunk_index(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            chunk_id = str(row.get("chunk_id") or "")
            if chunk_id:
                index[chunk_id] = row
    return index


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
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


if __name__ == "__main__":
    main()
