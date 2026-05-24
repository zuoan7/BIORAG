from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.audit_v3_retrieval_drift import (
    TOP_N,
    fmt,
    load_jsonl,
    parent_chunk_id,
    rank_in_chunks,
    write_json,
    write_jsonl,
    write_markdown,
)
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.schemas import RetrievedChunk
from src.synbio_rag.infrastructure.vectorstores.bm25 import _tokenize, tokenize_query
from src.synbio_rag.infrastructure.vectorstores.fusion import reciprocal_rank_fusion_multi


VARIANTS = ("b0_stable", "b1_parent_expansion")
RESULTS_ROOT = Path("results/evaluation")
REPORTS_ROOT = Path("reports/evaluation")
DEFAULT_DRIFT_AUDIT_DIR = RESULTS_ROOT / "v3_retrieval_drift_20260523_raw_child_trace_debug"
DEFAULT_DATASET_PATH = Path("data/eval/datasets/v3_baseline_dataset.jsonl")
RRF_K = 60


@dataclass(frozen=True)
class NormConfig:
    name: str
    query_mode: str
    augmented_index: bool = False
    caption_boost: float = 0.0


NORM_CONFIGS = (
    NormConfig("current", query_mode="original"),
    NormConfig("query_alias", query_mode="original_plus_alias"),
    NormConfig("english_mirror", query_mode="english_only"),
    NormConfig("english_mirror_plus_original", query_mode="original_plus_english"),
    NormConfig("english_mirror_plus_alias", query_mode="original_plus_english_alias"),
    NormConfig("caption_text_aug", query_mode="original_plus_alias", augmented_index=True),
    NormConfig("full_norm_boost", query_mode="original_plus_english_alias", augmented_index=True, caption_boost=0.08),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline v3 query/caption/table normalization audit.")
    parser.add_argument("--drift-audit-dir", default=str(DEFAULT_DRIFT_AUDIT_DIR))
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
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
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise SystemExit(f"dataset path not found: {dataset_path}")

    run_id = args.run_id or derive_run_id(drift_dir)
    output_dir = RESULTS_ROOT / f"v3_query_caption_norm_{run_id}"
    report_dir = REPORTS_ROOT / f"v3_query_caption_norm_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings.from_env()
    dataset_by_id = load_dataset_by_id(dataset_path)
    child_records = load_child_records(Path(settings.kb.child_chunk_jsonl))
    base_index = InMemoryBM25(child_records, settings.retrieval, augmented=False)
    augmented_index = InMemoryBM25(child_records, settings.retrieval, augmented=True)

    samples_by_variant: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    for variant in VARIANTS:
        rows = [row for row in load_jsonl(drift_dir / f"{variant}_samples.jsonl") if row.get("gold_child_chunk_ids")]
        if args.limit > 0:
            rows = rows[: args.limit]
        samples = [
            audit_sample(
                variant=variant,
                row=row,
                dataset_row=dataset_by_id.get(str(row.get("sample_id") or "")) or {},
                base_index=base_index,
                augmented_index=augmented_index,
            )
            for row in rows
        ]
        samples_by_variant[variant] = samples
        summaries[variant] = summarize_samples(samples)
        write_jsonl(output_dir / f"{variant}_samples.jsonl", samples)

    summary = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "drift_audit_dir": str(drift_dir),
        "dataset_path": str(dataset_path),
        "child_chunk_jsonl": settings.kb.child_chunk_jsonl,
        "scope": "previous retrieval drift audit samples with gold_child_chunk_ids present",
        "top_n": TOP_N,
        "llm_calls": 0,
        "english_mirror_source": "metadata.original_generated_question from dataset, fallback deterministic rules",
        "configs": [config_dict(config) for config in NORM_CONFIGS],
        "variants": summaries,
        "comparison": compare_variants(samples_by_variant),
    }
    write_json(output_dir / "audit_summary.json", summary)
    write_markdown(report_dir / "report.md", render_report(summary, samples_by_variant))
    print(json.dumps({"output_dir": str(output_dir), "report_dir": str(report_dir)}, ensure_ascii=False))


class InMemoryBM25:
    def __init__(self, records: list[RetrievedChunk], retrieval_config: Any, *, augmented: bool) -> None:
        self.records = records
        self.config = retrieval_config
        self.augmented = augmented
        self.doc_len: list[int] = []
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.term_freqs: list[Counter[str]] = []
        self.avgdl = 0.0
        self._build()

    def _build(self) -> None:
        for chunk in self.records:
            text = retrieval_text(chunk)
            if self.augmented:
                text = f"{text}\n{caption_text_aliases(chunk)}"
            terms = _tokenize(text)
            tf = Counter(terms)
            self.term_freqs.append(tf)
            self.doc_len.append(len(terms))
            for term in tf:
                self.doc_freq[term] += 1
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0

    def search(self, query: str, limit: int) -> list[RetrievedChunk]:
        query_terms = tokenize_query(query) or _tokenize(query)
        if not query_terms:
            return []
        scored: list[RetrievedChunk] = []
        for idx, chunk in enumerate(self.records):
            score = self._score(query_terms, idx)
            if score <= 0.0:
                continue
            scored.append(clone_chunk(chunk, bm25_score=score))
        scored.sort(key=lambda item: item.bm25_score, reverse=True)
        return scored[:limit]

    def _score(self, query_terms: list[str], doc_idx: int) -> float:
        score = 0.0
        tf = self.term_freqs[doc_idx]
        dl = self.doc_len[doc_idx]
        n_docs = len(self.records)
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            df = self.doc_freq.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = freq + self.config.bm25_k1 * (
                1 - self.config.bm25_b + self.config.bm25_b * dl / max(self.avgdl, 1.0)
            )
            score += idf * (freq * (self.config.bm25_k1 + 1)) / max(denom, 1e-9)
        return score


def audit_sample(
    *,
    variant: str,
    row: dict[str, Any],
    dataset_row: dict[str, Any],
    base_index: InMemoryBM25,
    augmented_index: InMemoryBM25,
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or "")
    question = str(row.get("question") or "")
    english_mirror = english_mirror_query(question, dataset_row)
    alias_query = query_aliases(question)
    gold_children = [str(item) for item in row.get("gold_child_chunk_ids") or []]
    gold_parents = [str(item) for item in row.get("gold_parent_chunk_ids") or []]
    wrong_child = str(((row.get("wrong_candidate_comparison") or {}).get("wrong") or {}).get("raw_child_chunk_id") or "")
    wrong_parent = str(((row.get("wrong_candidate_comparison") or {}).get("wrong") or {}).get("raw_parent_chunk_id") or "")

    config_results = {}
    for config in NORM_CONFIGS:
        queries = queries_for_config(
            config.query_mode,
            original=question,
            english=english_mirror,
            alias=alias_query,
        )
        index = augmented_index if config.augmented_index else base_index
        fused = run_sparse_rrf(index=index, queries=queries, caption_boost=config.caption_boost, top_n=TOP_N)
        gold_child_hit = rank_in_chunks(fused, set(gold_children), by_parent=False)
        gold_parent_hit = rank_in_chunks(fused, set(gold_parents), by_parent=True)
        wrong_child_hit = rank_in_chunks(fused, {wrong_child}, by_parent=False) if wrong_child else None
        wrong_parent_hit = rank_in_chunks(fused, {wrong_parent}, by_parent=True) if wrong_parent else None
        config_results[config.name] = {
            "queries": queries,
            "augmented_index": config.augmented_index,
            "caption_boost": config.caption_boost,
            "gold_child_rank": rank_or_none(gold_child_hit),
            "gold_parent_rank": rank_or_none(gold_parent_hit),
            "wrong_child_rank": rank_or_none(wrong_child_hit),
            "wrong_parent_rank": rank_or_none(wrong_parent_hit),
            "gold_child_hit_top40": hit_within(gold_child_hit, 40),
            "gold_child_hit_top80": hit_within(gold_child_hit, 80),
            "gold_child_hit_top200": hit_within(gold_child_hit, 200),
            "gold_parent_hit_top200": hit_within(gold_parent_hit, 200),
            "gold_before_wrong_child": before(gold_child_hit, wrong_child_hit),
            "gold_before_wrong_parent": before(gold_parent_hit, wrong_parent_hit),
            "top_hits": compact_hits(fused[:8]),
        }

    return {
        "variant_key": variant,
        "sample_id": sample_id,
        "category": row.get("category"),
        "question": question,
        "english_mirror_query": english_mirror,
        "query_alias": alias_query,
        "expected_doc_ids": [str(item) for item in row.get("expected_doc_ids") or []],
        "gold_child_chunk_ids": gold_children,
        "gold_parent_chunk_ids": gold_parents,
        "wrong_child_chunk_id": wrong_child,
        "wrong_parent_chunk_id": wrong_parent,
        "source_drift_classification": row.get("classification"),
        "diagnostic_tags": row.get("diagnostic_tags") or [],
        "config_results": config_results,
        "first_gold_child_rescue_config": first_rescue(config_results, key="gold_child_hit_top200"),
        "first_gold_parent_rescue_config": first_rescue(config_results, key="gold_parent_hit_top200"),
        "first_gold_before_wrong_config": first_gold_before_wrong(config_results),
    }


def run_sparse_rrf(
    *,
    index: InMemoryBM25,
    queries: list[str],
    caption_boost: float,
    top_n: int,
) -> list[RetrievedChunk]:
    sparse_runs = [(index.search(query, top_n), 1.0) for query in queries if query.strip()]
    fused = reciprocal_rank_fusion_multi(dense_runs=[], sparse_runs=sparse_runs, limit=top_n, rrf_k=RRF_K)
    if caption_boost <= 0:
        return fused
    boosted = []
    for chunk in fused:
        item = clone_chunk(chunk)
        if should_caption_boost(item, " ".join(queries)):
            item.fusion_score += caption_boost
        boosted.append(item)
    boosted.sort(key=lambda item: (item.fusion_score, item.bm25_score, item.chunk_id), reverse=True)
    return boosted[:top_n]


def queries_for_config(mode: str, *, original: str, english: str, alias: str) -> list[str]:
    queries = {
        "original": [original],
        "original_plus_alias": [original, alias],
        "english_only": [english],
        "original_plus_english": [original, english],
        "original_plus_english_alias": [original, english, alias],
    }.get(mode)
    if queries is None:
        raise ValueError(f"unknown query mode: {mode}")
    return dedupe([query for query in queries if query.strip()])


def english_mirror_query(question: str, dataset_row: dict[str, Any]) -> str:
    metadata = dataset_row.get("metadata") or {}
    original = str(metadata.get("original_generated_question") or "").strip()
    if original:
        return original
    return deterministic_english_mirror(question)


def deterministic_english_mirror(question: str) -> str:
    aliases = query_aliases(question)
    if aliases != question:
        return aliases
    return question


def query_aliases(question: str) -> str:
    text = str(question or "")
    aliases = [text]

    for fig, panel in re.findall(r"Figure\s*([0-9]+)\s*([A-Za-z])", text, flags=re.IGNORECASE):
        panel = panel.upper()
        aliases.extend([
            f"Figure {fig}{panel}",
            f"Figure {fig} {panel}",
            f"Figure {fig} ({panel})",
            f"panel {panel}",
            f"({panel})",
        ])
    for table in re.findall(r"Table\s*([S]?[0-9]+)", text, flags=re.IGNORECASE):
        aliases.extend([f"Table {table}", f"Table{table}", "table caption", "table footnote"])

    phrase_aliases = {
        "星号": ["*", "asterisk", "star", "legend", "marked", "labeled", "annotation"],
        "标注": ["marked", "labeled", "indicated", "annotation"],
        "细菌科": ["bacterial families", "family-level", "relative abundance"],
        "代表什么变化": ["altered", "changed", "restored toward control levels", "pretreatment"],
        "甲基化分析": ["methylation analysis"],
        "低聚糖": ["oligosaccharides"],
        "相对残基含量": ["relative amount", "residue", "linkage"],
        "分析方法": ["analytical method", "analysis method"],
        "计算": ["calculated", "based on", "total ion response", "GC-MS analysis"],
        "唾液酸转移酶": ["sialyltransferase"],
        "上调": ["upregulated", "up-regulated"],
        "转移": ["metastasis"],
        "化疗耐受": ["drug treatment resistance", "chemotherapy resistance"],
        "启动子": ["promoter"],
        "相互作用": ["interaction"],
        "酵母单杂交": ["yeast one-hybrid"],
        "Western blot": ["Western blot", "glycosylated", "unglycosylated"],
    }
    for phrase, values in phrase_aliases.items():
        if phrase in text:
            aliases.extend(values)
    if "Figure" in text or "图" in text:
        aliases.extend(["figure caption", "panel", "legend"])
    if "Table" in text or "表" in text:
        aliases.extend(["table caption", "table footnote"])
    return " ".join(dedupe(aliases))


def caption_text_aliases(chunk: RetrievedChunk) -> str:
    metadata = chunk.metadata or {}
    text = retrieval_text(chunk)
    aliases: list[str] = []
    has_figure = bool(metadata.get("contains_figure_caption") or "[FIGURE CAPTION]" in text)
    has_table = bool(metadata.get("contains_table_caption") or metadata.get("contains_table_text") or "[TABLE CAPTION]" in text)
    if has_figure:
        aliases.extend(["figure caption", "figure legend", "panel", "asterisk", "star", "marked", "labeled"])
        for fig in re.findall(r"Figure\s+([0-9]+)", text, flags=re.IGNORECASE):
            aliases.extend([f"Figure {fig}", f"Figure{fig}"])
            for panel in sorted(set(re.findall(r"\(([A-Z])\)", text))):
                aliases.extend([f"Figure {fig}{panel}", f"Figure {fig} {panel}", f"Figure {fig} ({panel})", f"panel {panel}"])
        if "*" in text:
            aliases.extend(["asterisk legend", "star legend", "significant annotation"])
    if has_table:
        aliases.extend(["table caption", "table footnote", "table legend", "calculated", "analysis method"])
        for table in re.findall(r"Table\s+([S]?[0-9]+)", text, flags=re.IGNORECASE):
            aliases.extend([f"Table {table}", f"Table{table}"])
        if "*" in text:
            aliases.extend(["asterisk footnote", "table footnote"])
        if "GC-MS" in text or "GC MS" in text:
            aliases.extend(["GC-MS analysis", "total ion response", "relative amount", "residue"])
    return " ".join(dedupe(aliases))


def should_caption_boost(chunk: RetrievedChunk, query: str) -> bool:
    metadata = chunk.metadata or {}
    lowered = query.lower()
    asks_figure = "figure" in lowered or "panel" in lowered or "asterisk" in lowered or "*" in query
    asks_table = "table" in lowered or "footnote" in lowered or "relative amount" in lowered
    if asks_figure and metadata.get("contains_figure_caption"):
        return True
    if asks_table and (metadata.get("contains_table_caption") or metadata.get("contains_table_text")):
        return True
    return False


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rescue_counts = {}
    parent_rescue_counts = {}
    before_wrong_counts = {}
    for config in NORM_CONFIGS:
        rescue_counts[config.name] = sum(
            1 for sample in samples if sample["config_results"][config.name]["gold_child_hit_top200"]
        )
        parent_rescue_counts[config.name] = sum(
            1 for sample in samples if sample["config_results"][config.name]["gold_parent_hit_top200"]
        )
        before_wrong_counts[config.name] = sum(
            1 for sample in samples if sample["config_results"][config.name]["gold_before_wrong_child"] is True
        )
    first_child = Counter(str(sample.get("first_gold_child_rescue_config") or "unresolved") for sample in samples)
    first_parent = Counter(str(sample.get("first_gold_parent_rescue_config") or "unresolved") for sample in samples)
    return {
        "target_case_count": len(samples),
        "gold_child_hit_top200_counts": rescue_counts,
        "gold_parent_hit_top200_counts": parent_rescue_counts,
        "gold_before_wrong_child_counts": before_wrong_counts,
        "first_gold_child_rescue_counts": dict(first_child),
        "first_gold_parent_rescue_counts": dict(first_parent),
        "unresolved_gold_child_sample_ids": [
            str(sample.get("sample_id") or "")
            for sample in samples
            if sample.get("first_gold_child_rescue_config") is None
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
        "# v3 Query/Caption Normalization 离线审计报告",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- drift_audit_dir: `{summary['drift_audit_dir']}`",
        f"- dataset_path: `{summary['dataset_path']}`",
        f"- scope: `{summary['scope']}`",
        f"- llm_calls: {summary['llm_calls']}",
        f"- english_mirror_source: {summary['english_mirror_source']}",
        "",
        "## Config Rescue Counts",
        "",
        "| config | query mode | augmented index | caption boost | b0 gold child@200 | b0 parent@200 | b0 gold before wrong | b1 gold child@200 | b1 parent@200 | b1 gold before wrong |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in NORM_CONFIGS:
        b0 = summary["variants"].get("b0_stable", {})
        b1 = summary["variants"].get("b1_parent_expansion", {})
        lines.append(
            f"| `{config.name}` | `{config.query_mode}` | {int(config.augmented_index)} | {config.caption_boost:g} | "
            f"{(b0.get('gold_child_hit_top200_counts') or {}).get(config.name, 0)} | "
            f"{(b0.get('gold_parent_hit_top200_counts') or {}).get(config.name, 0)} | "
            f"{(b0.get('gold_before_wrong_child_counts') or {}).get(config.name, 0)} | "
            f"{(b1.get('gold_child_hit_top200_counts') or {}).get(config.name, 0)} | "
            f"{(b1.get('gold_parent_hit_top200_counts') or {}).get(config.name, 0)} | "
            f"{(b1.get('gold_before_wrong_child_counts') or {}).get(config.name, 0)} |"
        )

    lines.extend(["", "## B0 Sample Detail", ""])
    lines.append("| sample_id | category | current child/wrong | english child/wrong | full child/wrong | first child rescue | first gold > wrong |")
    lines.append("|---|---|---|---|---|---|---|")
    for sample in samples_by_variant.get("b0_stable", []):
        current = sample["config_results"]["current"]
        english = sample["config_results"]["english_mirror_plus_original"]
        full = sample["config_results"]["full_norm_boost"]
        lines.append(
            f"| `{sample['sample_id']}` | {sample.get('category')} | "
            f"{fmt(current.get('gold_child_rank'))}/{fmt(current.get('wrong_child_rank'))} | "
            f"{fmt(english.get('gold_child_rank'))}/{fmt(english.get('wrong_child_rank'))} | "
            f"{fmt(full.get('gold_child_rank'))}/{fmt(full.get('wrong_child_rank'))} | "
            f"`{sample.get('first_gold_child_rescue_config') or 'unresolved'}` | "
            f"`{sample.get('first_gold_before_wrong_config') or 'unresolved'}` |"
        )

    lines.extend(["", "## English Mirror Queries", ""])
    for sample in samples_by_variant.get("b0_stable", []):
        lines.append(f"- `{sample['sample_id']}`: {sample['english_mirror_query']}")
    return "\n".join(lines)


def first_rescue(config_results: dict[str, dict[str, Any]], *, key: str) -> str | None:
    for config in NORM_CONFIGS:
        if (config_results.get(config.name) or {}).get(key):
            return config.name
    return None


def first_gold_before_wrong(config_results: dict[str, dict[str, Any]]) -> str | None:
    for config in NORM_CONFIGS:
        if (config_results.get(config.name) or {}).get("gold_before_wrong_child") is True:
            return config.name
    return None


def load_dataset_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id") or ""): row for row in load_jsonl(path)}


def load_child_records(path: Path) -> list[RetrievedChunk]:
    records = []
    for item in load_jsonl(path):
        metadata = {
            "chunk_index": item.get("chunk_index"),
            "retrieval_text": item.get("retrieval_text", ""),
            "content_kind": item.get("content_kind", "body"),
            "quality_score": item.get("quality_score", 0.0),
            "contains_table_text": item.get("contains_table_text", False),
            "contains_table_caption": item.get("contains_table_caption", False),
            "contains_figure_caption": item.get("contains_figure_caption", False),
            "contains_image": item.get("contains_image", False),
            "object_type": item.get("object_type", "body"),
            "object_id": item.get("object_id", ""),
            "block_types": item.get("block_types", []),
            "block_ids": item.get("block_ids", []),
            "source_block_ids": item.get("source_block_ids", []),
            "evidence_types": item.get("evidence_types", []),
            "page_numbers": item.get("page_numbers", []),
            "section_path": item.get("section_path", []),
            "index_role": item.get("index_role", ""),
            "parent_id": item.get("parent_id", ""),
            "parent_chunk_id": item.get("parent_chunk_id", ""),
            "child_index": item.get("child_index"),
        }
        records.append(
            RetrievedChunk(
                chunk_id=item.get("chunk_id", ""),
                doc_id=item.get("doc_id", ""),
                source_file=item.get("source_file", ""),
                title=item.get("title", ""),
                section=item.get("section", ""),
                text=item.get("text", ""),
                page_start=item.get("page_start"),
                page_end=item.get("page_end"),
                metadata=metadata,
            )
        )
    return records


def retrieval_text(chunk: RetrievedChunk) -> str:
    metadata = chunk.metadata or {}
    return str(metadata.get("retrieval_text") or chunk.text or "")


def clone_chunk(chunk: RetrievedChunk, *, bm25_score: float | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        source_file=chunk.source_file,
        title=chunk.title,
        section=chunk.section,
        text=chunk.text,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        vector_score=chunk.vector_score,
        bm25_score=chunk.bm25_score if bm25_score is None else bm25_score,
        rerank_score=chunk.rerank_score,
        fusion_score=chunk.fusion_score,
        metadata=dict(chunk.metadata),
    )


def rank_or_none(hit: dict[str, Any] | None) -> int | None:
    if hit is None:
        return None
    try:
        return int(hit["rank"])
    except (KeyError, TypeError, ValueError):
        return None


def hit_within(hit: dict[str, Any] | None, limit: int) -> bool:
    rank = rank_or_none(hit)
    return bool(rank is not None and rank <= limit)


def before(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool | None:
    left_rank = rank_or_none(left)
    right_rank = rank_or_none(right)
    if left_rank is None:
        return False if right_rank is not None else None
    if right_rank is None:
        return True
    return left_rank < right_rank


def compact_hits(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    return [
        {
            "rank": idx,
            "chunk_id": chunk.chunk_id,
            "parent_chunk_id": parent_chunk_id(chunk.chunk_id),
            "doc_id": chunk.doc_id,
            "section": chunk.section,
            "bm25_score": round(float(chunk.bm25_score or 0.0), 6),
            "fusion_score": round(float(chunk.fusion_score or 0.0), 6),
            "block_types": (chunk.metadata or {}).get("block_types") or [],
        }
        for idx, chunk in enumerate(chunks, start=1)
    ]


def config_dict(config: NormConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "query_mode": config.query_mode,
        "augmented_index": config.augmented_index,
        "caption_boost": config.caption_boost,
    }


def derive_run_id(drift_dir: Path) -> str:
    prefix = "v3_retrieval_drift_"
    name = drift_dir.name
    if name.startswith(prefix):
        return name[len(prefix):]
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def dedupe(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def run_self_test() -> None:
    q = "Figure 9C 中用星号标注的细菌科代表什么变化？"
    alias = query_aliases(q)
    assert "Figure 9 (C)" in alias, alias
    assert "asterisk" in alias, alias
    assert "bacterial families" in alias, alias
    table_alias = query_aliases("甲基化分析低聚糖 4 和 5 时，相对残基含量用什么分析方法计算？")
    assert "methylation analysis" in table_alias, table_alias
    assert "GC-MS analysis" in table_alias, table_alias
    dataset_row = {"metadata": {"original_generated_question": "What method was used?"}}
    assert english_mirror_query("中文问题", dataset_row) == "What method was used?"
    print("self-test passed")


if __name__ == "__main__":
    main()
