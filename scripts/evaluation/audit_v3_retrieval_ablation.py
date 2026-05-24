from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.audit_v3_retrieval_drift import (
    BASELINE_RETRIEVAL_OVERRIDES,
    TOP_N,
    RetrievalProber,
    apply_overrides,
    best_rank_across_runs,
    fmt,
    load_jsonl,
    rank_in_chunks,
    round_number,
    topn_presence,
    write_json,
    write_jsonl,
    write_markdown,
)
from src.synbio_rag.application.retrieval_postprocessor import RetrievalPostProcessor
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.infrastructure.vectorstores.fusion import reciprocal_rank_fusion_multi


VARIANTS = ("b0_stable", "b1_parent_expansion")
RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
DEFAULT_DRIFT_AUDIT_DIR = RESULTS_ROOT / "v3_retrieval_drift_20260523_raw_child_trace_debug"
DEFAULT_RAW_LIMIT = 40


@dataclass(frozen=True)
class AblationConfig:
    name: str
    source_limit: int
    fusion_pool: int
    cjk_bm25_weight: float


ABLATION_CONFIGS = (
    AblationConfig("current", source_limit=40, fusion_pool=120, cjk_bm25_weight=0.25),
    AblationConfig("source_80", source_limit=80, fusion_pool=120, cjk_bm25_weight=0.25),
    AblationConfig("source_120", source_limit=120, fusion_pool=120, cjk_bm25_weight=0.25),
    AblationConfig("source_200", source_limit=200, fusion_pool=120, cjk_bm25_weight=0.25),
    AblationConfig("fusion_200", source_limit=200, fusion_pool=200, cjk_bm25_weight=0.25),
    AblationConfig("fusion_400", source_limit=200, fusion_pool=400, cjk_bm25_weight=0.25),
    AblationConfig("bm25_weight_0_5", source_limit=200, fusion_pool=400, cjk_bm25_weight=0.5),
    AblationConfig("bm25_weight_1_0", source_limit=200, fusion_pool=400, cjk_bm25_weight=1.0),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline v3 retrieval ablation audit.")
    parser.add_argument("--drift-audit-dir", default=str(DEFAULT_DRIFT_AUDIT_DIR))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    drift_dir = Path(args.drift_audit_dir)
    if not drift_dir.exists():
        raise SystemExit(f"drift audit dir not found: {drift_dir}")

    run_id = args.run_id or derive_run_id(drift_dir)
    output_dir = RESULTS_ROOT / f"v3_retrieval_ablation_{run_id}"
    report_dir = REPORTS_ROOT / f"v3_retrieval_ablation_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings.from_env()
    apply_overrides(settings, BASELINE_RETRIEVAL_OVERRIDES)
    prober = RetrievalProber(settings)
    postprocessor = RetrievalPostProcessor()

    probe_cache: dict[str, dict[str, Any]] = {}
    samples_by_variant: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}

    for variant in VARIANTS:
        rows = load_jsonl(drift_dir / f"{variant}_samples.jsonl")
        targets = [row for row in rows if row.get("gold_child_chunk_ids")]
        if args.limit > 0:
            targets = targets[: args.limit]

        samples: list[dict[str, Any]] = []
        for row in targets:
            question = str(row.get("question") or "")
            if question not in probe_cache:
                probe = prober.probe(question, top_n=TOP_N)
                probe["analysis_object"] = prober.router.analyze(question)
                probe["contains_cjk"] = prober.query_planner.contains_cjk(question)
                probe_cache[question] = probe
            samples.append(
                audit_sample(
                    variant=variant,
                    row=row,
                    probe=probe_cache[question],
                    retrieval_config=settings.retrieval,
                    postprocessor=postprocessor,
                )
            )
        samples_by_variant[variant] = samples
        summaries[variant] = summarize_samples(samples)
        write_jsonl(output_dir / f"{variant}_samples.jsonl", samples)

    summary = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "drift_audit_dir": str(drift_dir),
        "child_chunk_jsonl": settings.kb.child_chunk_jsonl,
        "parent_chunk_jsonl": settings.kb.parent_chunk_jsonl,
        "parent_index_path": settings.retrieval.parent_index_path,
        "dense_index_id_kind": prober.dense_id_kind,
        "scope": "previous retrieval drift audit samples with gold_child_chunk_ids present",
        "top_n": TOP_N,
        "default_raw_limit": DEFAULT_RAW_LIMIT,
        "ablation_configs": [config_dict(config) for config in ABLATION_CONFIGS],
        "variants": summaries,
        "comparison": compare_variants(samples_by_variant),
    }
    write_json(output_dir / "audit_summary.json", summary)
    write_markdown(report_dir / "report.md", render_report(summary, samples_by_variant))
    print(json.dumps({"output_dir": str(output_dir), "report_dir": str(report_dir)}, ensure_ascii=False))


def audit_sample(
    *,
    variant: str,
    row: dict[str, Any],
    probe: dict[str, Any],
    retrieval_config: Any,
    postprocessor: RetrievalPostProcessor,
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or "")
    question = str(row.get("question") or "")
    gold_children = [str(item) for item in row.get("gold_child_chunk_ids") or []]
    gold_parents = [str(item) for item in row.get("gold_parent_chunk_ids") or []]

    dense_parent = best_rank_across_runs(probe["dense_runs"], set(gold_parents), by_parent=True)
    bm25_child = best_rank_across_runs(probe["sparse_runs"], set(gold_children), by_parent=False)

    ablation_results: dict[str, dict[str, Any]] = {}
    for config in ABLATION_CONFIGS:
        ablation_results[config.name] = simulate_ablation(
            config=config,
            row=row,
            probe=probe,
            retrieval_config=retrieval_config,
            postprocessor=postprocessor,
            gold_children=gold_children,
            gold_parents=gold_parents,
        )

    first_rescue = first_rescue_config(ablation_results)
    first_drop_stage, recommended_next_bucket = classify_ablation_outcome(
        first_rescue=first_rescue,
        ablation_results=ablation_results,
        dense_parent=dense_parent,
        bm25_child=bm25_child,
        diagnostic_tags=[str(item) for item in row.get("diagnostic_tags") or []],
    )

    return {
        "variant_key": variant,
        "sample_id": sample_id,
        "category": row.get("category"),
        "question": question,
        "expected_doc_ids": [str(item) for item in row.get("expected_doc_ids") or []],
        "gold_child_chunk_ids": gold_children,
        "gold_parent_chunk_ids": gold_parents,
        "source_drift_classification": row.get("classification"),
        "source_drift_drop_stage": row.get("drop_stage"),
        "diagnostic_tags": row.get("diagnostic_tags") or [],
        "dense_gold_parent_proxy": topn_presence(dense_parent),
        "bm25_gold_child_exact": topn_presence(bm25_child),
        "dense_gold_parent_proxy_rank_top200": rank_or_none(dense_parent),
        "bm25_gold_child_exact_rank_top200": rank_or_none(bm25_child),
        "ablation_results": ablation_results,
        "first_rescue_config": first_rescue,
        "first_drop_stage": first_drop_stage,
        "recommended_next_bucket": recommended_next_bucket,
        "wrong_candidate_comparison": row.get("wrong_candidate_comparison") or {},
    }


def simulate_ablation(
    *,
    config: AblationConfig,
    row: dict[str, Any],
    probe: dict[str, Any],
    retrieval_config: Any,
    postprocessor: RetrievalPostProcessor,
    gold_children: list[str],
    gold_parents: list[str],
) -> dict[str, Any]:
    question = str(row.get("question") or "")
    dense_runs = []
    sparse_runs = []
    query_variants = [item for item in probe.get("query_variants") or [] if isinstance(item, dict)]
    contains_cjk = bool(probe.get("contains_cjk"))

    for idx, variant in enumerate(query_variants):
        dense_chunks = list(probe["dense_runs"][idx][0])[: config.source_limit]
        sparse_chunks = list(probe["sparse_runs"][idx][0])[: config.source_limit]
        dense_weight = retrieval_config.dense_rrf_weight * float(variant.get("weight") or 0.0)
        bm25_weight = retrieval_config.bm25_rrf_weight
        if contains_cjk:
            bm25_weight *= config.cjk_bm25_weight
        sparse_weight = bm25_weight * float(variant.get("weight") or 0.0)
        dense_runs.append((dense_chunks, dense_weight))
        sparse_runs.append((sparse_chunks, sparse_weight))

    fused = reciprocal_rank_fusion_multi(
        dense_runs=dense_runs,
        sparse_runs=sparse_runs,
        limit=config.fusion_pool,
        rrf_k=retrieval_config.rrf_k,
    )
    fusion_parent = rank_in_chunks(fused, set(gold_parents), by_parent=True)
    fusion_child = rank_in_chunks(fused, set(gold_children), by_parent=False)

    boosted = postprocessor.apply_title_keyword_boost(fused, question, retrieval_config)
    boosted = postprocessor.apply_structure_marker_boost(boosted, question, retrieval_config)
    boosted_parent = rank_in_chunks(boosted, set(gold_parents), by_parent=True)
    boosted_child = rank_in_chunks(boosted, set(gold_children), by_parent=False)

    diversified = postprocessor.apply_comparison_diversity(
        boosted,
        DEFAULT_RAW_LIMIT,
        probe.get("analysis_object"),
        retrieval_config,
    )
    diversified_parent = rank_in_chunks(diversified, set(gold_parents), by_parent=True)
    diversified_child = rank_in_chunks(diversified, set(gold_children), by_parent=False)

    first_dense = dense_runs[0][0] if dense_runs else []
    first_sparse = sparse_runs[0][0] if sparse_runs else []
    expanded = postprocessor.apply_same_doc_body_expansion(
        diversified=diversified,
        dense_results=first_dense,
        bm25_results=first_sparse,
        config=retrieval_config,
        question=question,
        bm25_retriever=None,
    )
    expanded = postprocessor.apply_source_floor(
        expanded=expanded,
        dense_results=first_dense,
        sparse_results=first_sparse,
        config=retrieval_config,
    )

    raw_parent = rank_in_chunks(expanded, set(gold_parents), by_parent=True)
    raw_child = rank_in_chunks(expanded, set(gold_children), by_parent=False)

    return {
        "source_limit": config.source_limit,
        "fusion_pool": config.fusion_pool,
        "cjk_bm25_weight": config.cjk_bm25_weight,
        "raw_gold_parent_hit": raw_parent is not None,
        "raw_gold_child_exact_hit": raw_child is not None,
        "rescued": raw_parent is not None or raw_child is not None,
        "gold_parent_rank": rank_or_none(raw_parent),
        "gold_child_rank": rank_or_none(raw_child),
        "fusion_gold_parent_rank": rank_or_none(fusion_parent),
        "fusion_gold_child_rank": rank_or_none(fusion_child),
        "boosted_gold_parent_rank": rank_or_none(boosted_parent),
        "boosted_gold_child_rank": rank_or_none(boosted_child),
        "default_raw_gold_parent_rank": rank_or_none(diversified_parent),
        "default_raw_gold_child_rank": rank_or_none(diversified_child),
        "raw_result_count": len(expanded),
        "same_doc_body_expand_enabled": bool(getattr(retrieval_config, "same_doc_body_expand_enabled", False)),
        "source_floor_enabled": bool(getattr(retrieval_config, "source_floor_enabled", False)),
    }


def classify_ablation_outcome(
    *,
    first_rescue: str | None,
    ablation_results: dict[str, dict[str, Any]],
    dense_parent: dict[str, Any] | None,
    bm25_child: dict[str, Any] | None,
    diagnostic_tags: list[str],
) -> tuple[str, str]:
    if first_rescue in {"source_80", "source_120", "source_200"}:
        return "source_limit", "source_depth_candidate"
    if first_rescue in {"fusion_200", "fusion_400"}:
        return "fusion_pool", "fusion_pool_candidate"
    if first_rescue in {"bm25_weight_0_5", "bm25_weight_1_0"}:
        return "fusion_pool", "bm25_weight_candidate"
    if first_rescue == "current":
        return "not_dropped_in_current_simulation", "source_depth_candidate"

    strongest = ablation_results.get("bm25_weight_1_0") or {}
    fusion_has_gold = (
        strongest.get("fusion_gold_parent_rank") is not None
        or strongest.get("fusion_gold_child_rank") is not None
    )
    boosted_has_gold = (
        strongest.get("boosted_gold_parent_rank") is not None
        or strongest.get("boosted_gold_child_rank") is not None
    )
    if fusion_has_gold or boosted_has_gold:
        return "default_raw_limit", "fusion_pool_candidate"
    if dense_parent is not None or bm25_child is not None:
        return "fusion_pool", "fusion_pool_candidate"
    if "query_text_mismatch" in diagnostic_tags or "chunk_boundary_mismatch" in diagnostic_tags:
        return "query_or_boundary_unresolved", "boundary_or_query_review"
    return "query_or_boundary_unresolved", "boundary_or_query_review"


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rescue_counts: dict[str, int] = {}
    for config in ABLATION_CONFIGS:
        rescue_counts[config.name] = sum(
            1 for sample in samples if sample["ablation_results"][config.name]["rescued"]
        )
    first_rescue_counts = Counter(
        str(sample.get("first_rescue_config") or "unresolved") for sample in samples
    )
    stage_counts = Counter(str(sample.get("first_drop_stage") or "unknown") for sample in samples)
    bucket_counts = Counter(str(sample.get("recommended_next_bucket") or "unknown") for sample in samples)
    return {
        "target_case_count": len(samples),
        "dense_gold_parent_proxy_top200_count": sum(
            1 for sample in samples if sample.get("dense_gold_parent_proxy_rank_top200") is not None
        ),
        "bm25_gold_child_exact_top200_count": sum(
            1 for sample in samples if sample.get("bm25_gold_child_exact_rank_top200") is not None
        ),
        "rescue_counts": rescue_counts,
        "first_rescue_counts": dict(first_rescue_counts),
        "first_drop_stage_counts": dict(stage_counts),
        "recommended_next_bucket_counts": dict(bucket_counts),
        "unresolved_count": first_rescue_counts.get("unresolved", 0),
        "unresolved_sample_ids": [
            str(sample.get("sample_id") or "")
            for sample in samples
            if sample.get("first_rescue_config") is None
        ],
    }


def compare_variants(samples_by_variant: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    b0_ids = {str(sample.get("sample_id") or "") for sample in samples_by_variant.get("b0_stable", [])}
    b1_ids = {str(sample.get("sample_id") or "") for sample in samples_by_variant.get("b1_parent_expansion", [])}
    return {
        "same_sample_ids": b0_ids == b1_ids,
        "shared_sample_count": len(b0_ids & b1_ids),
        "b0_only": sorted(b0_ids - b1_ids),
        "b1_only": sorted(b1_ids - b0_ids),
    }


def render_report(summary: dict[str, Any], samples_by_variant: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# v3 Retrieval Ablation 审计报告",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- drift_audit_dir: `{summary['drift_audit_dir']}`",
        f"- scope: `{summary['scope']}`",
        f"- dense_index_id_kind: `{summary['dense_index_id_kind']}`",
        f"- top_n: {summary['top_n']}",
        f"- default_raw_limit: {summary['default_raw_limit']}",
        "",
        "## 关键说明",
        "",
        "- 本脚本只做离线 debug/audit；未调用 judge，未重跑 B0/B1 end-to-end eval，未修改检索、rerank、support/citation 或答案生成。",
        "- dense/Milvus 以 parent/base chunk id 作为有效 proxy；BM25 child exact 来自 child chunk 缓存。",
        "- source depth、fusion pool、CJK BM25 weight 是离线复放参数；same-doc body expansion 仅按当前配置执行。",
        "",
        "## Rescue Count",
        "",
        "| config | source limit | fusion pool | CJK BM25 weight | b0 rescued | b0 first rescue | b1 rescued | b1 first rescue |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in ABLATION_CONFIGS:
        b0 = summary["variants"].get("b0_stable", {})
        b1 = summary["variants"].get("b1_parent_expansion", {})
        lines.append(
            f"| `{config.name}` | {config.source_limit} | {config.fusion_pool} | "
            f"{config.cjk_bm25_weight:g} | "
            f"{(b0.get('rescue_counts') or {}).get(config.name, 0)} | "
            f"{(b0.get('first_rescue_counts') or {}).get(config.name, 0)} | "
            f"{(b1.get('rescue_counts') or {}).get(config.name, 0)} | "
            f"{(b1.get('first_rescue_counts') or {}).get(config.name, 0)} |"
        )

    lines.extend(["", "## First Rescue / Unresolved", ""])
    lines.append("| sample_id | dense parent@200 | BM25 child@200 | first rescue | drop stage | next bucket | current parent/child | strongest parent/child |")
    lines.append("|---|---:|---:|---|---|---|---|---|")
    for sample in samples_by_variant.get("b0_stable", []):
        current = sample["ablation_results"]["current"]
        strongest = sample["ablation_results"]["bm25_weight_1_0"]
        lines.append(
            f"| `{sample['sample_id']}` | {fmt(sample.get('dense_gold_parent_proxy_rank_top200'))} | "
            f"{fmt(sample.get('bm25_gold_child_exact_rank_top200'))} | "
            f"`{sample.get('first_rescue_config') or 'unresolved'}` | "
            f"`{sample['first_drop_stage']}` | `{sample['recommended_next_bucket']}` | "
            f"{fmt(current.get('gold_parent_rank'))}/{fmt(current.get('gold_child_rank'))} | "
            f"{fmt(strongest.get('gold_parent_rank'))}/{fmt(strongest.get('gold_child_rank'))} |"
        )

    lines.extend(["", "## Stage Summary", ""])
    lines.append("| 变体 | target | dense parent@200 | BM25 child@200 | unresolved | stage counts | next bucket counts |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for variant, item in summary["variants"].items():
        lines.append(
            f"| {variant} | {item['target_case_count']} | "
            f"{item['dense_gold_parent_proxy_top200_count']} | "
            f"{item['bm25_gold_child_exact_top200_count']} | "
            f"{item['unresolved_count']} | "
            f"{json.dumps(item['first_drop_stage_counts'], ensure_ascii=False, sort_keys=True)} | "
            f"{json.dumps(item['recommended_next_bucket_counts'], ensure_ascii=False, sort_keys=True)} |"
        )

    lines.extend(["", "## v3_pc_055 / v3_pc_078 Unresolved Evidence", ""])
    for sample_id in ("v3_pc_055", "v3_pc_078"):
        sample = find_sample(samples_by_variant.get("b0_stable", []), sample_id)
        if sample is None:
            lines.append(f"- `{sample_id}`: not present in this run.")
            continue
        comparison = sample.get("wrong_candidate_comparison") or {}
        gold_overlap = ((comparison.get("gold") or {}).get("overlap") or {}).get("overlap_ratio")
        wrong_overlap = ((comparison.get("wrong") or {}).get("overlap") or {}).get("overlap_ratio")
        strongest = sample["ablation_results"]["bm25_weight_1_0"]
        lines.append(
            f"- `{sample_id}`: first_rescue=`{sample.get('first_rescue_config') or 'unresolved'}`, "
            f"drop_stage=`{sample['first_drop_stage']}`, tags={sample.get('diagnostic_tags') or []}, "
            f"dense parent@200={fmt(sample.get('dense_gold_parent_proxy_rank_top200'))}, "
            f"BM25 child@200={fmt(sample.get('bm25_gold_child_exact_rank_top200'))}, "
            f"strongest raw parent/child={fmt(strongest.get('gold_parent_rank'))}/{fmt(strongest.get('gold_child_rank'))}, "
            f"fusion parent/child={fmt(strongest.get('fusion_gold_parent_rank'))}/{fmt(strongest.get('fusion_gold_child_rank'))}, "
            f"nearest wrong-parent distance={fmt(comparison.get('parent_distance'))}, "
            f"overlap gold/wrong={fmt(gold_overlap)}/{fmt(wrong_overlap)}."
        )
    return "\n".join(lines)


def first_rescue_config(ablation_results: dict[str, dict[str, Any]]) -> str | None:
    for config in ABLATION_CONFIGS:
        if (ablation_results.get(config.name) or {}).get("rescued"):
            return config.name
    return None


def rank_or_none(hit: dict[str, Any] | None) -> int | None:
    if hit is None:
        return None
    try:
        return int(hit["rank"])
    except (KeyError, TypeError, ValueError):
        return None


def config_dict(config: AblationConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "source_limit": config.source_limit,
        "fusion_pool": config.fusion_pool,
        "cjk_bm25_weight": config.cjk_bm25_weight,
    }


def find_sample(samples: list[dict[str, Any]], sample_id: str) -> dict[str, Any] | None:
    for sample in samples:
        if str(sample.get("sample_id") or "") == sample_id:
            return sample
    return None


def derive_run_id(drift_dir: Path) -> str:
    prefix = "v3_retrieval_drift_"
    name = drift_dir.name
    if name.startswith(prefix):
        return name[len(prefix):]
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_self_test() -> None:
    assert derive_run_id(Path("results/evaluation/v3_retrieval_drift_abc")) == "abc"
    assert first_rescue_config({"current": {"rescued": False}, "source_80": {"rescued": True}}) == "source_80"
    stage, bucket = classify_ablation_outcome(
        first_rescue="source_120",
        ablation_results={},
        dense_parent={"rank": 90},
        bm25_child=None,
        diagnostic_tags=[],
    )
    assert (stage, bucket) == ("source_limit", "source_depth_candidate")
    stage, bucket = classify_ablation_outcome(
        first_rescue=None,
        ablation_results={"bm25_weight_1_0": {"fusion_gold_parent_rank": None, "fusion_gold_child_rank": None}},
        dense_parent=None,
        bm25_child=None,
        diagnostic_tags=["chunk_boundary_mismatch"],
    )
    assert (stage, bucket) == ("query_or_boundary_unresolved", "boundary_or_query_review")
    assert round_number("1.23456789") == 1.234568
    print("self-test passed")


if __name__ == "__main__":
    main()
