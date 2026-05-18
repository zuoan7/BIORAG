#!/usr/bin/env python3
"""Run retrieval-only Phase 4E-0 sanity probes.

This script uses the existing HybridRetriever with Milvus dense retrieval and
BM25 cache retrieval. It does not call an LLM, generator, or reranker.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synbio_rag.domain.config import KnowledgeBaseConfig, RetrievalConfig
from src.synbio_rag.infrastructure.embedding.bge import BGEM3Embedder
from src.synbio_rag.infrastructure.vectorstores.bm25 import BM25Retriever
from src.synbio_rag.infrastructure.vectorstores.hybrid import HybridRetriever
from src.synbio_rag.infrastructure.vectorstores.milvus import MilvusRetriever


MARKER_RE = re.compile(r"\[(?:TABLE CAPTION|TABLE TEXT|FIGURE CAPTION)\]\s*")
FIGURE_NUMBER_RE = re.compile(r"\b(?:fig(?:ure)?\.?)\s*([A-Za-z]?\d+[A-Za-z]?)\b", re.I)
TABLE_NUMBER_RE = re.compile(r"\btable\s*([A-Za-z]?\d+[A-Za-z]?)\b", re.I)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'._+-]{2,}|\d+(?:\.\d+)?%?")
STOPWORDS = {
    "about",
    "analysis",
    "and",
    "are",
    "caption",
    "cell",
    "cells",
    "does",
    "figure",
    "from",
    "into",
    "show",
    "shown",
    "shows",
    "study",
    "table",
    "that",
    "the",
    "these",
    "this",
    "using",
    "what",
    "which",
    "with",
}
LOW_INFORMATION_TERMS = STOPWORDS | {
    "continued",
    "been",
    "being",
    "fig",
    "figs",
    "most",
    "supplemental",
    "supplementary",
    "there",
    "used",
    "was",
    "were",
}

NUMBER_ONLY_CAPTION_PATTERN = re.compile(
    r"^\s*(?:table|fig(?:ure)?\.?)\s+s?[A-Za-z]?\d+[A-Za-z]?[.:|() ]*\s*$",
    re.I,
)
CONTINUED_ONLY_CAPTION_PATTERN = re.compile(
    r"^\s*(?:table|fig(?:ure)?\.?)\s+s?[A-Za-z]?\d+[A-Za-z]?[.:|() ]*"
    r"(?:continued|cont\.?)\s*$",
    re.I,
)
GENERIC_TABLE_CAPTION_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in (
        r"\bstrains?\s+(?:and|,)\s+plasmids?\s+used\s+in\s+(?:this|the)\s+study\b",
        r"\bstrains?,\s+genes?,\s+and\s+plasmids?\s+used\s+in\s+(?:this|the)\s+study\b",
        r"\bprimers?\s+used\s+in\s+(?:this|the)\s+study\b",
        r"\bsupplementary\s+table\s+s?\d+[.:]?\s*\d?\.?\s*$",
    )
]
GENERIC_PROBE_QUERY_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in (
        r"^what\s+does\s+table\s+s?\d+\s+report\??$",
        r"^what\s+does\s+figure\s+s?\d+\s+show\??$",
        r"^which\s+figure\s+shows\s+fig\??$",
        r"^what\s+is\s+shown\s+in\s+figure\s+s?\d+\s+about\s+fig\??$",
        r"^which\s+table\s+reports\s+continued\??$",
        r"^which\s+table\s+reports\s+supplementary\??$",
    )
]

FALSE_TABLE_CAPTION_PATTERN = re.compile(
    r"^\s*(?:\[TABLE CAPTION\]\s*)?"
    r"table\s+s?\d+[.:]?\s+(?:the\s+)?[A-Z]\.?\s*$",
    re.I,
)
FALSE_FIGURE_CAPTION_PATTERN = re.compile(
    r"^\s*(?:\[FIGURE CAPTION\]\s*)?(?:fig(?:ure)?\.?)\s+s?\d+[A-Z]?[.:]?\s*$",
    re.I,
)

REQUIRED_DOC0367_QUERIES = [
    "What does Figure 5 compare in doc_0367?",
    "What does the figure show about Opto-T7RNAP and paT7P-148?",
    "Which figure compares Opto-T7RNAPs to paT7P-148?",
]


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
            if isinstance(item, dict):
                chunks.append(item)
    return chunks


def is_table_focused(chunk: dict[str, Any]) -> bool:
    return bool(chunk.get("contains_table_caption") or chunk.get("contains_table_text"))


def is_figure_focused(chunk: dict[str, Any]) -> bool:
    return bool(chunk.get("contains_figure_caption"))


def is_evidence_chunk(chunk: dict[str, Any]) -> bool:
    return is_table_focused(chunk) or is_figure_focused(chunk)


def is_caption_only_table(chunk: dict[str, Any]) -> bool:
    return bool(chunk.get("contains_table_caption") and not chunk.get("contains_table_text"))


def is_short_caption(chunk: dict[str, Any]) -> bool:
    return is_evidence_chunk(chunk) and len(clean_body(chunk.get("text", ""))) < 80


def is_likely_false_caption(chunk: dict[str, Any]) -> bool:
    body = clean_body(chunk.get("text", ""))
    if chunk.get("contains_table_caption") and FALSE_TABLE_CAPTION_PATTERN.match(body):
        return True
    if chunk.get("contains_figure_caption") and FALSE_FIGURE_CAPTION_PATTERN.match(body):
        return True
    return False


def is_doc0367_figure5_chunk(chunk: dict[str, Any]) -> bool:
    return (
        chunk.get("doc_id") == "doc_0367"
        and is_figure_focused(chunk)
        and re.search(r"\bfig(?:ure)?\.?\s*5\b", clean_body(chunk.get("text", "")), re.I)
        is not None
    )


def risk_slices_for_chunk(chunk: dict[str, Any]) -> list[str]:
    slices: list[str] = []
    if is_evidence_chunk(chunk) and chunk.get("section") == "Title":
        slices.append("section_title_evidence")
    if is_short_caption(chunk):
        slices.append("short_captions")
    if is_likely_false_caption(chunk):
        slices.append("likely_false_caption")
    if is_caption_only_table(chunk):
        slices.append("caption_only_tables")
    if is_figure_focused(chunk):
        slices.append("figure_captions")
    if not is_evidence_chunk(chunk) and int(chunk.get("token_count") or 0) >= 80:
        slices.append("paragraph_heavy_normal_controls")
    if is_doc0367_figure5_chunk(chunk):
        slices.append("doc_0367_figure5")
    return slices


def evidence_type_from_chunk_obj(chunk: Any) -> str:
    metadata = getattr(chunk, "metadata", {}) or {}
    table = bool(metadata.get("contains_table_caption") or metadata.get("contains_table_text"))
    figure = bool(metadata.get("contains_figure_caption"))
    if table and figure:
        return "table_figure"
    if table:
        return "table"
    if figure:
        return "figure"
    return "paragraph"


def clean_body(text: Any) -> str:
    body = MARKER_RE.sub("", str(text or ""))
    return re.sub(r"\s+", " ", body).strip()


def keyword_phrase(text: str, max_terms: int = 5) -> str:
    tokens = []
    for token in TOKEN_RE.findall(text):
        lowered = token.lower().strip(".,;:()[]{}")
        if len(lowered) < 3 or lowered in STOPWORDS:
            continue
        if lowered not in tokens:
            tokens.append(lowered)
        if len(tokens) >= max_terms:
            break
    return " ".join(tokens)


def discriminative_terms(text: str, max_terms: int = 8) -> list[str]:
    terms = []
    for token in TOKEN_RE.findall(text):
        lowered = token.lower().strip(".,;:()[]{}")
        if len(lowered) < 3 or lowered in LOW_INFORMATION_TERMS:
            continue
        if lowered.replace(".", "").isdigit():
            continue
        if lowered not in terms:
            terms.append(lowered)
        if len(terms) >= max_terms:
            break
    return terms


def figure_number(text: str) -> str:
    match = FIGURE_NUMBER_RE.search(text)
    return match.group(1) if match else ""


def table_number(text: str) -> str:
    match = TABLE_NUMBER_RE.search(text)
    return match.group(1) if match else ""


def sorted_sample(chunks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return sorted(chunks, key=lambda item: (str(item.get("doc_id", "")), str(item.get("chunk_id", ""))))[:limit]


def make_table_probe(chunk: dict[str, Any], variant: str = "table_number") -> dict[str, Any]:
    body = clean_body(chunk.get("text", ""))
    number = table_number(body)
    phrase = keyword_phrase(body)
    if variant == "keyword" and phrase:
        query = f"Which table reports {phrase}?"
    elif number and phrase:
        query = f"What does Table {number} report about {phrase}?"
    elif number:
        query = f"What does Table {number} report?"
    else:
        query = f"Which table reports {phrase}?"
    return make_probe("table", query, chunk, subtype=f"table_{variant}")


def make_figure_probe(chunk: dict[str, Any], variant: str = "figure_number") -> dict[str, Any]:
    body = clean_body(chunk.get("text", ""))
    number = figure_number(body)
    phrase = keyword_phrase(body)
    if variant == "keyword" and phrase:
        query = f"Which figure shows {phrase}?"
    elif number and phrase:
        query = f"What is shown in Figure {number} about {phrase}?"
    elif number:
        query = f"What does Figure {number} show?"
    else:
        query = f"What is shown in the figure about {phrase}?"
    return make_probe("figure", query, chunk, subtype=f"figure_{variant}")


def clean_keyword_phrase(text: str, max_terms: int = 5) -> str:
    return " ".join(discriminative_terms(text, max_terms=max_terms))


def make_clean_table_probe(chunk: dict[str, Any], variant: str = "table_number") -> dict[str, Any]:
    body = clean_body(chunk.get("text", ""))
    number = table_number(body)
    phrase = clean_keyword_phrase(body)
    if variant == "keyword" and phrase:
        query = f"Which table reports {phrase}?"
    elif number and phrase:
        query = f"What does Table {number} report about {phrase}?"
    elif number:
        query = f"What does Table {number} report?"
    else:
        query = f"Which table reports {phrase}?"
    return make_probe("table", query, chunk, subtype=f"table_{variant}")


def make_clean_figure_probe(chunk: dict[str, Any], variant: str = "figure_number") -> dict[str, Any]:
    body = clean_body(chunk.get("text", ""))
    number = figure_number(body)
    phrase = clean_keyword_phrase(body)
    if variant == "keyword" and phrase:
        query = f"Which figure shows {phrase}?"
    elif number and phrase:
        query = f"What is shown in Figure {number} about {phrase}?"
    elif number:
        query = f"What does Figure {number} show?"
    else:
        query = f"What is shown in the figure about {phrase}?"
    return make_probe("figure", query, chunk, subtype=f"figure_{variant}")


def make_normal_probe(chunk: dict[str, Any]) -> dict[str, Any]:
    body = clean_body(chunk.get("text", ""))
    phrase = keyword_phrase(body, max_terms=6)
    if not phrase:
        phrase = str(chunk.get("title") or chunk.get("section") or "the paragraph")
    query = f"What does the paper report about {phrase}?"
    return make_probe("normal", query, chunk, subtype="normal_factoid")


def make_probe(
    probe_type: str,
    query: str,
    chunk: dict[str, Any],
    subtype: str = "",
) -> dict[str, Any]:
    return {
        "type": probe_type,
        "subtype": subtype,
        "query": query,
        "target_doc_id": chunk.get("doc_id", ""),
        "target_chunk_id": chunk.get("chunk_id", ""),
        "target_section": chunk.get("section", ""),
        "target_text_preview": clean_body(chunk.get("text", ""))[:300],
        "risk_slices": risk_slices_for_chunk(chunk),
    }


def make_doc0367_probes(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        chunk
        for chunk in chunks
        if is_doc0367_figure5_chunk(chunk)
    ]
    if not candidates:
        return []
    target = sorted_sample(candidates, 1)[0]
    probes = []
    for query in REQUIRED_DOC0367_QUERIES:
        probe = make_probe("figure", query, target, subtype="doc_0367_figure5")
        probe["subtype"] = "doc_0367_figure5"
        probes.append(probe)
    return probes


def caption_exclusion_reasons(chunk: dict[str, Any], probe_type: str) -> list[str]:
    body = clean_body(chunk.get("text", ""))
    terms = discriminative_terms(body)
    reasons: list[str] = []
    if is_likely_false_caption(chunk):
        reasons.append("likely_false_caption")
    if NUMBER_ONLY_CAPTION_PATTERN.match(body):
        reasons.append("number_only_caption")
    if CONTINUED_ONLY_CAPTION_PATTERN.match(body):
        reasons.append("continued_only_caption")
    if probe_type == "table" and any(pattern.search(body) for pattern in GENERIC_TABLE_CAPTION_PATTERNS):
        reasons.append("generic_table_caption")
    if len(terms) < 2:
        reasons.append("insufficient_discriminative_caption_terms")
    return sorted(set(reasons))


def probe_exclusion_reasons(probe: dict[str, Any]) -> list[str]:
    query = clean_body(probe.get("query", ""))
    reasons = [
        "generic_probe_query"
        for pattern in GENERIC_PROBE_QUERY_PATTERNS
        if pattern.match(query)
    ]
    if probe.get("type") in {"table", "figure"} and len(discriminative_terms(query)) < 2:
        reasons.append("insufficient_discriminative_query_terms")
    return sorted(set(reasons))


def probe_quality_example(probe: dict[str, Any], reasons: list[str] | None = None) -> dict[str, Any]:
    item = {
        "type": probe.get("type", ""),
        "subtype": probe.get("subtype", ""),
        "query": probe.get("query", ""),
        "target_doc_id": probe.get("target_doc_id", ""),
        "target_chunk_id": probe.get("target_chunk_id", ""),
        "target_preview": probe.get("target_text_preview", ""),
        "risk_slices": probe.get("risk_slices", []),
    }
    if reasons is not None:
        item["reasons"] = reasons
    return item


def add_excluded_probe(
    quality_report: dict[str, Any],
    probe_type: str,
    probe: dict[str, Any],
    reasons: list[str],
) -> None:
    section = quality_report[probe_type]
    section["excluded_probe_count"] += 1
    for reason in reasons:
        section["excluded_by_reason"][reason] += 1
    if len(section["excluded_examples"]) < 20:
        section["excluded_examples"].append(probe_quality_example(probe, reasons))


def add_selected_probe(
    quality_report: dict[str, Any],
    probe_type: str,
    probe: dict[str, Any],
) -> None:
    section = quality_report[probe_type]
    section["selected_gating_probe_count"] += 1
    if len(section["selected_examples"]) < 10:
        section["selected_examples"].append(probe_quality_example(probe))


def init_probe_quality_report(probe_policy: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "probe_policy": probe_policy,
        "table": {
            "candidate_chunk_count": sum(1 for chunk in chunks if is_table_focused(chunk)),
            "selected_gating_probe_count": 0,
            "excluded_probe_count": 0,
            "excluded_by_reason": Counter(),
            "selected_examples": [],
            "excluded_examples": [],
        },
        "figure": {
            "candidate_chunk_count": sum(1 for chunk in chunks if is_figure_focused(chunk)),
            "selected_gating_probe_count": 0,
            "excluded_probe_count": 0,
            "excluded_by_reason": Counter(),
            "selected_examples": [],
            "excluded_examples": [],
        },
        "normal": {
            "selected_probe_count": 0,
        },
    }


def normalize_probe_quality_report(report: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(report)
    for probe_type in ("table", "figure"):
        section = dict(normalized[probe_type])
        section["excluded_by_reason"] = dict(sorted(section["excluded_by_reason"].items()))
        normalized[probe_type] = section
    return normalized


def build_clean_table_figure_probes(
    chunks: list[dict[str, Any]],
    sample_per_type: int,
    quality_report: dict[str, Any],
) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for probe_type, candidates, maker in (
        ("table", [chunk for chunk in chunks if is_table_focused(chunk)], make_clean_table_probe),
        ("figure", [chunk for chunk in chunks if is_figure_focused(chunk)], make_clean_figure_probe),
    ):
        for chunk in sorted_sample(candidates, len(candidates)):
            chunk_id = str(chunk.get("chunk_id", ""))
            if not chunk_id or chunk_id in seen:
                continue
            variant = "keyword" if quality_report[probe_type]["selected_gating_probe_count"] % 2 else (
                "table_number" if probe_type == "table" else "figure_number"
            )
            probe = maker(chunk, variant=variant)
            reasons = caption_exclusion_reasons(chunk, probe_type) + probe_exclusion_reasons(probe)
            reasons = sorted(set(reasons))
            if reasons:
                add_excluded_probe(quality_report, probe_type, probe, reasons)
                continue
            probes.append(probe)
            seen.add(chunk_id)
            add_selected_probe(quality_report, probe_type, probe)
            if quality_report[probe_type]["selected_gating_probe_count"] >= sample_per_type:
                break
    existing = {(probe["query"], probe["target_chunk_id"]) for probe in probes}
    for probe in make_doc0367_probes(chunks):
        key = (probe["query"], probe["target_chunk_id"])
        if key in existing:
            continue
        reasons = caption_exclusion_reasons(
            {
                "text": probe["target_text_preview"],
                "contains_figure_caption": True,
                "doc_id": probe["target_doc_id"],
                "chunk_id": probe["target_chunk_id"],
            },
            "figure",
        ) + probe_exclusion_reasons(probe)
        if reasons:
            add_excluded_probe(quality_report, "figure", probe, sorted(set(reasons)))
            continue
        probes.append(probe)
        add_selected_probe(quality_report, "figure", probe)
        existing.add(key)
    return probes


def build_probes_with_report(
    chunks: list[dict[str, Any]],
    sample_per_type: int,
    mode: str,
    probe_policy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if probe_policy == "legacy":
        probes = build_probes(chunks, sample_per_type=sample_per_type, mode=mode)
        return probes, {
            "probe_policy": probe_policy,
            "selected_probe_count": len(probes),
            "selected_by_type": dict(Counter(probe["type"] for probe in probes)),
        }

    quality_report = init_probe_quality_report(probe_policy, chunks)
    probes: list[dict[str, Any]] = []
    if mode in ("table_figure_probe", "both"):
        probes.extend(build_clean_table_figure_probes(chunks, sample_per_type, quality_report))
    if mode in ("normal_probe", "both"):
        normal_probes = [make_normal_probe(chunk) for chunk in select_normal_chunks(chunks, sample_per_type)]
        probes.extend(normal_probes)
        quality_report["normal"]["selected_probe_count"] = len(normal_probes)
    return probes, normalize_probe_quality_report(quality_report)


def add_unique_chunks(
    selected: list[dict[str, Any]],
    seen: set[str],
    candidates: list[dict[str, Any]],
    limit: int,
) -> None:
    for chunk in sorted_sample(candidates, len(candidates)):
        if len(selected) >= limit:
            return
        chunk_id = str(chunk.get("chunk_id", ""))
        if not chunk_id or chunk_id in seen:
            continue
        selected.append(chunk)
        seen.add(chunk_id)


def select_table_chunks(chunks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    table_chunks = [chunk for chunk in chunks if is_table_focused(chunk)]
    add_unique_chunks(selected, seen, [chunk for chunk in table_chunks if chunk.get("section") == "Title"], limit)
    add_unique_chunks(selected, seen, [chunk for chunk in table_chunks if is_likely_false_caption(chunk)], limit)
    add_unique_chunks(selected, seen, [chunk for chunk in table_chunks if is_short_caption(chunk)], limit)
    add_unique_chunks(selected, seen, [chunk for chunk in table_chunks if is_caption_only_table(chunk)], limit)
    add_unique_chunks(selected, seen, table_chunks, limit)
    return selected[:limit]


def select_figure_chunks(chunks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    figure_chunks = [chunk for chunk in chunks if is_figure_focused(chunk)]
    add_unique_chunks(selected, seen, [chunk for chunk in figure_chunks if is_doc0367_figure5_chunk(chunk)], limit)
    add_unique_chunks(selected, seen, [chunk for chunk in figure_chunks if chunk.get("section") == "Title"], limit)
    add_unique_chunks(selected, seen, [chunk for chunk in figure_chunks if is_likely_false_caption(chunk)], limit)
    add_unique_chunks(selected, seen, [chunk for chunk in figure_chunks if is_short_caption(chunk)], limit)
    add_unique_chunks(selected, seen, figure_chunks, limit)
    return selected[:limit]


def select_normal_chunks(chunks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    evidence_doc_ids = {
        str(chunk.get("doc_id", ""))
        for chunk in chunks
        if is_evidence_chunk(chunk)
    }
    paragraph_chunks = [
        chunk
        for chunk in chunks
        if not is_evidence_chunk(chunk) and int(chunk.get("token_count") or 0) >= 80
    ]
    add_unique_chunks(
        selected,
        seen,
        [chunk for chunk in paragraph_chunks if str(chunk.get("doc_id", "")) in evidence_doc_ids],
        max(limit // 2, 1),
    )
    add_unique_chunks(
        selected,
        seen,
        [chunk for chunk in paragraph_chunks if str(chunk.get("doc_id", "")) not in evidence_doc_ids],
        limit,
    )
    add_unique_chunks(selected, seen, paragraph_chunks, limit)
    if len(selected) < limit:
        add_unique_chunks(
            selected,
            seen,
            [chunk for chunk in chunks if not is_evidence_chunk(chunk)],
            limit,
        )
    return selected[:limit]


def build_probes(chunks: list[dict[str, Any]], sample_per_type: int, mode: str) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    if mode in ("table_figure_probe", "both"):
        table_chunks = select_table_chunks(chunks, sample_per_type)
        figure_chunks = select_figure_chunks(chunks, sample_per_type)
        for idx, chunk in enumerate(table_chunks):
            variant = "keyword" if idx % 2 else "table_number"
            probes.append(make_table_probe(chunk, variant=variant))
        for idx, chunk in enumerate(figure_chunks):
            variant = "keyword" if idx % 2 else "figure_number"
            probes.append(make_figure_probe(chunk, variant=variant))
        existing = {(probe["query"], probe["target_chunk_id"]) for probe in probes}
        for probe in make_doc0367_probes(chunks):
            key = (probe["query"], probe["target_chunk_id"])
            if key not in existing:
                probes.append(probe)
                existing.add(key)
    if mode in ("normal_probe", "both"):
        probes.extend(make_normal_probe(chunk) for chunk in select_normal_chunks(chunks, sample_per_type))
    return probes


def build_retriever(
    chunks_jsonl: Path,
    milvus_uri: str,
    collection_name: str,
    bm25_index_path: Path,
    model_path: str,
    top_k: int,
) -> tuple[dict[str, Any], int]:
    retrieval_config = RetrievalConfig(
        milvus_uri=milvus_uri,
        collection_name=collection_name,
        bm25_cache_path=str(bm25_index_path),
        dense_limit=max(40, top_k),
        bm25_limit=max(40, top_k),
        search_limit=max(40, top_k),
        score_floor=0.0,
    )
    kb_config = KnowledgeBaseConfig(
        chunk_jsonl=str(chunks_jsonl),
        embedding_model_path=model_path,
        embedding_max_length=512,
    )
    embedder = BGEM3Embedder(
        model_path=kb_config.embedding_model_path,
        dim=kb_config.embedding_dim,
        max_length=kb_config.embedding_max_length,
    )
    dense = MilvusRetriever(retrieval_config, embedder)
    bm25 = BM25Retriever(retrieval_config, kb_config, milvus_client=dense.client)
    bm25._ensure_index()
    hybrid = HybridRetriever(retrieval_config, dense, bm25)
    return {"dense_only": dense, "bm25_only": bm25, "hybrid": hybrid}, len(bm25._records)


def serialize_hit(hit: Any, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "section": hit.section,
        "evidence_type": evidence_type_from_chunk_obj(hit),
        "fusion_score": hit.fusion_score,
        "vector_score": hit.vector_score,
        "bm25_score": hit.bm25_score,
        "text_preview": clean_body(hit.text)[:240],
    }


def run_probe(retriever: Any, probe: dict[str, Any], top_k: int) -> dict[str, Any]:
    hits = retriever.search(probe["query"], limit=top_k)
    serialized = [serialize_hit(hit, idx) for idx, hit in enumerate(hits[:top_k], start=1)]
    target_chunk = probe["target_chunk_id"]
    target_doc = probe["target_doc_id"]
    chunk_rank = next((hit["rank"] for hit in serialized if hit["chunk_id"] == target_chunk), None)
    doc_rank = next((hit["rank"] for hit in serialized if hit["doc_id"] == target_doc), None)
    result = dict(probe)
    result.update(
        {
            "doc_hit": doc_rank is not None,
            "chunk_hit": chunk_rank is not None,
            "doc_rank": doc_rank,
            "chunk_rank": chunk_rank,
            "top_k": serialized,
        }
    )
    return result


def metric_summary(results: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_type[result["type"]].append(result)
    summary: dict[str, Any] = {}
    for probe_type, items in sorted(by_type.items()):
        total = len(items)
        doc_hits = sum(1 for item in items if item["doc_hit"])
        chunk_hits = sum(1 for item in items if item["chunk_hit"])
        summary[probe_type] = {
            "count": total,
            f"doc_hit@{top_k}": doc_hits,
            f"chunk_hit@{top_k}": chunk_hits,
            f"doc_hit_rate@{top_k}": doc_hits / total if total else 0.0,
            f"chunk_hit_rate@{top_k}": chunk_hits / total if total else 0.0,
        }
    return summary


def topk_distribution(results: list[dict[str, Any]]) -> dict[str, Any]:
    distribution: dict[str, Any] = {}
    for result in results:
        probe_type = result["type"]
        if probe_type not in distribution:
            distribution[probe_type] = {
                "topk_evidence_type_counts": Counter(),
                "query_count": 0,
                "topk_total": 0,
                "table_figure_topk_total": 0,
            }
        entry = distribution[probe_type]
        entry["query_count"] += 1
        for hit in result["top_k"]:
            evidence_type = hit["evidence_type"]
            entry["topk_evidence_type_counts"][evidence_type] += 1
            entry["topk_total"] += 1
            if evidence_type in {"table", "figure", "table_figure"}:
                entry["table_figure_topk_total"] += 1
    normalized: dict[str, Any] = {}
    for probe_type, entry in distribution.items():
        total = entry["topk_total"]
        normalized[probe_type] = {
            "query_count": entry["query_count"],
            "topk_total": total,
            "topk_evidence_type_counts": dict(sorted(entry["topk_evidence_type_counts"].items())),
            "table_figure_topk_total": entry["table_figure_topk_total"],
            "table_figure_topk_rate": entry["table_figure_topk_total"] / total if total else 0.0,
        }
    return normalized


def risk_slice_summary(results: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        for slice_name in result.get("risk_slices", []):
            by_slice[slice_name].append(result)
    summary: dict[str, Any] = {}
    for slice_name, items in sorted(by_slice.items()):
        total = len(items)
        doc_hits = sum(1 for item in items if item["doc_hit"])
        chunk_hits = sum(1 for item in items if item["chunk_hit"])
        distribution = topk_distribution(items)
        summary[slice_name] = {
            "count": total,
            f"doc_hit@{top_k}": doc_hits,
            f"chunk_hit@{top_k}": chunk_hits,
            f"doc_hit_rate@{top_k}": doc_hits / total if total else 0.0,
            f"chunk_hit_rate@{top_k}": chunk_hits / total if total else 0.0,
            "topk_distribution": distribution,
        }
    return summary


def display_score(hit: dict[str, Any]) -> float:
    return float(hit.get("fusion_score") or hit.get("vector_score") or hit.get("bm25_score") or 0.0)


def write_miss_examples(path: Path, results_by_mode: dict[str, list[dict[str, Any]]], top_k: int) -> None:
    hybrid_misses = [result for result in results_by_mode.get("hybrid", []) if not result["chunk_hit"]]
    total_misses = {
        mode: sum(1 for result in results if not result["chunk_hit"])
        for mode, results in sorted(results_by_mode.items())
    }
    lines = [
        "# Phase 4E-1 Miss Examples",
        "",
        f"- total_chunk_misses@{top_k}: `{json.dumps(total_misses, ensure_ascii=False)}`",
        f"- hybrid_chunk_misses@{top_k}: {len(hybrid_misses)}",
        "",
    ]
    ordered: list[tuple[str, dict[str, Any]]] = [("hybrid", result) for result in hybrid_misses]
    for mode, results in sorted(results_by_mode.items()):
        if mode == "hybrid":
            continue
        ordered.extend((mode, result) for result in results if not result["chunk_hit"])
    for mode, result in ordered[:60]:
        lines.extend(
            [
                f"## {mode} | {result['type']} | {result.get('subtype', '')} | {result['target_chunk_id']}",
                "",
                f"- query: {result['query']}",
                f"- target_doc_id: `{result['target_doc_id']}`",
                f"- doc_hit: `{result['doc_hit']}` rank: `{result['doc_rank']}`",
                f"- chunk_hit: `{result['chunk_hit']}` rank: `{result['chunk_rank']}`",
                f"- risk_slices: `{result.get('risk_slices', [])}`",
                f"- target_preview: {result['target_text_preview']}",
                "",
                "Top-k sample:",
            ]
        )
        for hit in result["top_k"][:5]:
            lines.append(
                f"- #{hit['rank']} `{hit['doc_id']}` `{hit['chunk_id']}` "
                f"{hit['evidence_type']} score={display_score(hit):.4f} :: {hit['text_preview']}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_excluded_probe_examples(path: Path, probe_quality_report: dict[str, Any]) -> None:
    lines = ["# Phase 4E-2 Excluded Probe Examples", ""]
    for probe_type in ("table", "figure"):
        section = probe_quality_report.get(probe_type, {})
        lines.extend(
            [
                f"## {probe_type}",
                "",
                f"- candidate_chunk_count: {section.get('candidate_chunk_count', 0)}",
                f"- selected_gating_probe_count: {section.get('selected_gating_probe_count', 0)}",
                f"- excluded_probe_count: {section.get('excluded_probe_count', 0)}",
                f"- excluded_by_reason: `{json.dumps(section.get('excluded_by_reason', {}), ensure_ascii=False)}`",
                "",
            ]
        )
        for item in section.get("excluded_examples", []):
            lines.extend(
                [
                    f"### {item.get('target_chunk_id', '')}",
                    "",
                    f"- reasons: `{item.get('reasons', [])}`",
                    f"- query: {item.get('query', '')}",
                    f"- target_doc_id: `{item.get('target_doc_id', '')}`",
                    f"- risk_slices: `{item.get('risk_slices', [])}`",
                    f"- target_preview: {item.get('target_preview', '')}",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    path: Path,
    metrics_by_mode: dict[str, Any],
    distribution_by_mode: dict[str, Any],
    risk_by_mode: dict[str, Any],
    doc0367_by_mode: dict[str, list[dict[str, Any]]],
    inputs: dict[str, Any],
    probe_quality_report: dict[str, Any],
) -> None:
    top_k = inputs["top_k"]
    hybrid_metrics = metrics_by_mode.get("hybrid", {})
    table_metrics = hybrid_metrics.get("table", {})
    figure_metrics = hybrid_metrics.get("figure", {})
    normal_metrics = hybrid_metrics.get("normal", {})
    hybrid_distribution = distribution_by_mode.get("hybrid", {})
    normal_rate = hybrid_distribution.get("normal", {}).get("table_figure_topk_rate", 0.0)
    doc0367_results = doc0367_by_mode.get("hybrid", [])
    doc0367_chunk_hits = sum(1 for result in doc0367_results if result.get("chunk_hit"))
    phase_pass = (
        table_metrics.get(f"doc_hit_rate@{top_k}", 0.0) >= 0.80
        and figure_metrics.get(f"doc_hit_rate@{top_k}", 0.0) >= 0.85
        and normal_metrics.get(f"doc_hit_rate@{top_k}", 0.0) >= 0.80
        and doc0367_chunk_hits >= 2
        and normal_rate <= 0.15
    )
    dense_table = metrics_by_mode.get("dense_only", {}).get("table", {})
    bm25_table = metrics_by_mode.get("bm25_only", {}).get("table", {})
    dense_diag = (
        "BM25 compensates"
        if dense_table.get(f"doc_hit_rate@{top_k}", 1.0) < 0.80
        and table_metrics.get(f"doc_hit_rate@{top_k}", 0.0) >= 0.80
        else "no blocker"
    )
    bm25_diag = (
        "dense compensates"
        if bm25_table.get(f"doc_hit_rate@{top_k}", 1.0) < 0.80
        and table_metrics.get(f"doc_hit_rate@{top_k}", 0.0) >= 0.80
        else "no blocker"
    )
    probe_policy = inputs.get("probe_policy", "legacy")
    phase_label = "Phase 4E-2 Clean Eval Probes" if probe_policy == "clean_gating" else "Phase 4E-1 Full Compact"
    pass_label = "phase4e2_pass" if probe_policy == "clean_gating" else "phase4e1_pass"
    lines = [
        f"# {phase_label} Retrieval-only Summary",
        "",
        "## Inputs",
        "",
        f"- chunks_jsonl: `{inputs['chunks_jsonl']}`",
        f"- milvus_uri: `{inputs['milvus_uri']}`",
        f"- collection_name: `{inputs['collection_name']}`",
        f"- bm25_index_path: `{inputs['bm25_index_path']}`",
        f"- top_k: {top_k}",
        f"- mode: `{inputs['mode']}`",
        f"- retrieval_modes: `{','.join(inputs['retrieval_modes'])}`",
        f"- probe_policy: `{probe_policy}`",
        f"- expected_chunk_count: {inputs['expected_chunk_count']}",
        f"- bm25_record_count: {inputs['bm25_record_count']}",
        "",
        "## Probe Quality",
        "",
    ]
    if probe_policy == "clean_gating":
        for probe_type in ("table", "figure"):
            section = probe_quality_report.get(probe_type, {})
            lines.extend(
                [
                    f"### {probe_type}",
                    f"- candidate_chunk_count: {section.get('candidate_chunk_count', 0)}",
                    f"- selected_gating_probe_count: {section.get('selected_gating_probe_count', 0)}",
                    f"- excluded_probe_count: {section.get('excluded_probe_count', 0)}",
                    f"- excluded_by_reason: `{json.dumps(section.get('excluded_by_reason', {}), ensure_ascii=False)}`",
                    "",
                ]
            )
        lines.extend(
            [
                f"- normal_selected_probe_count: {probe_quality_report.get('normal', {}).get('selected_probe_count', 0)}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- selected_probe_count: {probe_quality_report.get('selected_probe_count', 0)}",
                f"- selected_by_type: `{json.dumps(probe_quality_report.get('selected_by_type', {}), ensure_ascii=False)}`",
                "",
            ]
        )
    lines.extend(
        [
        "## Hit Metrics",
        "",
        ]
    )
    for retrieval_mode, metrics in metrics_by_mode.items():
        lines.extend([f"### {retrieval_mode}", ""])
        for probe_type in ("table", "figure", "normal"):
            item = metrics.get(probe_type, {})
            lines.extend(
                [
                    f"- {probe_type}: count={item.get('count', 0)} "
                    f"doc_hit@{top_k}={item.get(f'doc_hit@{top_k}', 0)} "
                    f"chunk_hit@{top_k}={item.get(f'chunk_hit@{top_k}', 0)} "
                    f"doc_rate={item.get(f'doc_hit_rate@{top_k}', 0.0):.3f} "
                    f"chunk_rate={item.get(f'chunk_hit_rate@{top_k}', 0.0):.3f}",
                ]
            )
        lines.append("")
    lines.extend(["## doc_0367 Figure 5", ""])
    for retrieval_mode, items in doc0367_by_mode.items():
        lines.append(f"### {retrieval_mode}")
        if not items:
            lines.append("- no doc_0367 Figure 5 probe target found")
        for result in items:
            lines.append(
                f"- `{result['query']}` doc_hit=`{result['doc_hit']}` "
                f"doc_rank=`{result['doc_rank']}` chunk_hit=`{result['chunk_hit']}` "
                f"chunk_rank=`{result['chunk_rank']}`"
            )
        lines.append("")
    lines.extend(["", "## Top-k Evidence Type Distribution", ""])
    for retrieval_mode, distribution in distribution_by_mode.items():
        lines.append(f"### {retrieval_mode}")
        for probe_type, item in sorted(distribution.items()):
            lines.append(f"- {probe_type}: `{json.dumps(item, ensure_ascii=False)}`")
        lines.append("")
    lines.extend(["## Risk Slices", ""])
    for retrieval_mode, risk_summary_by_slice in risk_by_mode.items():
        lines.append(f"### {retrieval_mode}")
        for slice_name, item in sorted(risk_summary_by_slice.items()):
            lines.append(
                f"- {slice_name}: count={item.get('count', 0)} "
                f"doc_rate={item.get(f'doc_hit_rate@{top_k}', 0.0):.3f} "
                f"chunk_rate={item.get(f'chunk_hit_rate@{top_k}', 0.0):.3f}"
            )
        lines.append("")
    lines.extend(
        [
            "",
            "## Normal Probe Table/Figure Occupancy",
            "",
            f"- table_figure_topk_rate: {normal_rate:.3f}",
            f"- hybrid_gate_threshold: 0.150",
            f"- fail_threshold: 0.350",
            f"- abnormal_takeover: `{normal_rate > 0.35}`",
            "",
            "## Risk Judgement",
            "",
            f"- compact_retrieval_text_abnormal_recall: `{normal_rate > 0.15}`",
            f"- table_figure_takeover_for_normal_queries: `{normal_rate > 0.15}`",
            f"- caption_only_table_basic_retrievable: `{risk_by_mode.get('hybrid', {}).get('caption_only_tables', {}).get(f'doc_hit_rate@{top_k}', 0.0) >= 0.80}`",
            f"- figure_caption_basic_retrievable: `{risk_by_mode.get('hybrid', {}).get('figure_captions', {}).get(f'doc_hit_rate@{top_k}', 0.0) >= 0.85}`",
            f"- section_title_evidence_exposed_blocker: `{risk_by_mode.get('hybrid', {}).get('section_title_evidence', {}).get(f'doc_hit_rate@{top_k}', 1.0) < 0.50}`",
            f"- parser_false_caption_sanity_blocker: `{risk_by_mode.get('hybrid', {}).get('likely_false_caption', {}).get(f'doc_hit_rate@{top_k}', 1.0) < 0.50}`",
            f"- dense_only_diagnostic: `{dense_diag}`",
            f"- bm25_only_diagnostic: `{bm25_diag}`",
            "",
            "## Decision",
            "",
            f"- {pass_label}: `{phase_pass}`",
            f"- recommend_formal_retrieval_benchmark_or_smoke_eval: `{phase_pass}`",
            f"- recommend_compact_for_later_main_chain_index_build_without_changing_defaults: `{phase_pass}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks_jsonl", required=True)
    parser.add_argument("--milvus_uri", required=True)
    parser.add_argument("--collection_name", required=True)
    parser.add_argument("--bm25_index_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--sample_per_type", type=int, default=20)
    parser.add_argument("--mode", choices=("table_figure_probe", "normal_probe", "both"), default="both")
    parser.add_argument(
        "--retrieval_mode",
        choices=("dense_only", "bm25_only", "hybrid"),
        default=None,
        help="Single retrieval mode to run. Prefer --retrieval_modes for Phase 4E-1.",
    )
    parser.add_argument(
        "--retrieval_modes",
        default="hybrid",
        help="Comma-separated retrieval modes: dense_only,bm25_only,hybrid",
    )
    parser.add_argument(
        "--probe_policy",
        choices=("legacy", "clean_gating"),
        default="legacy",
        help="legacy keeps Phase 4E-1 probe behavior; clean_gating filters low-information table/figure eval probes.",
    )
    parser.add_argument("--model_path", default=str(REPO_ROOT / "models/BAAI/bge-m3"))
    args = parser.parse_args()

    chunks_jsonl = Path(args.chunks_jsonl)
    bm25_index_path = Path(args.bm25_index_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bm25_index_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(chunks_jsonl)
    expected_chunk_count = len(chunks)
    probes, probe_quality_report = build_probes_with_report(
        chunks,
        sample_per_type=args.sample_per_type,
        mode=args.mode,
        probe_policy=args.probe_policy,
    )
    retrievers, bm25_record_count = build_retriever(
        chunks_jsonl=chunks_jsonl,
        milvus_uri=args.milvus_uri,
        collection_name=args.collection_name,
        bm25_index_path=bm25_index_path,
        model_path=args.model_path,
        top_k=args.top_k,
    )
    if bm25_record_count != expected_chunk_count:
        raise RuntimeError(
            f"BM25 record_count mismatch: {bm25_record_count} != {expected_chunk_count}"
        )

    if args.retrieval_mode:
        retrieval_modes = [args.retrieval_mode]
    else:
        retrieval_modes = [
            item.strip()
            for item in args.retrieval_modes.split(",")
            if item.strip()
        ]
    invalid_modes = [mode for mode in retrieval_modes if mode not in retrievers]
    if invalid_modes:
        raise ValueError(f"Unsupported retrieval mode(s): {invalid_modes}")

    results_by_mode: dict[str, list[dict[str, Any]]] = {}
    metrics_by_mode: dict[str, Any] = {}
    distribution_by_mode: dict[str, Any] = {}
    risk_by_mode: dict[str, Any] = {}
    doc0367_by_mode: dict[str, list[dict[str, Any]]] = {}
    for retrieval_mode in retrieval_modes:
        retriever = retrievers[retrieval_mode]
        results = [run_probe(retriever, probe, args.top_k) for probe in probes]
        results_by_mode[retrieval_mode] = results
        metrics_by_mode[retrieval_mode] = metric_summary(results, args.top_k)
        distribution_by_mode[retrieval_mode] = topk_distribution(results)
        risk_by_mode[retrieval_mode] = risk_slice_summary(results, args.top_k)
        doc0367_by_mode[retrieval_mode] = [
            result for result in results if result.get("subtype") == "doc_0367_figure5"
        ]

    inputs = {
        "chunks_jsonl": str(chunks_jsonl),
        "milvus_uri": args.milvus_uri,
        "collection_name": args.collection_name,
        "bm25_index_path": str(bm25_index_path),
        "top_k": args.top_k,
        "sample_per_type": args.sample_per_type,
        "mode": args.mode,
        "retrieval_modes": retrieval_modes,
        "probe_policy": args.probe_policy,
        "model_path": args.model_path,
        "bm25_index_exists": bm25_index_path.exists(),
        "expected_chunk_count": expected_chunk_count,
        "bm25_record_count": bm25_record_count,
    }
    payload = {
        "inputs": inputs,
        "metrics_by_mode": metrics_by_mode,
        "doc0367_figure5_results_by_mode": doc0367_by_mode,
        "probe_count": len(probes),
        "probe_quality_report": probe_quality_report,
        "results_by_mode": results_by_mode,
    }
    (output_dir / "retrieval_sanity_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "topk_distribution.json").write_text(
        json.dumps(distribution_by_mode, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "risk_slice_results.json").write_text(
        json.dumps(risk_by_mode, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "probe_quality_report.json").write_text(
        json.dumps(probe_quality_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_miss_examples(output_dir / "miss_examples.md", results_by_mode, args.top_k)
    write_excluded_probe_examples(output_dir / "excluded_probe_examples.md", probe_quality_report)
    write_summary(
        output_dir / "summary.md",
        metrics_by_mode,
        distribution_by_mode,
        risk_by_mode,
        doc0367_by_mode,
        inputs,
        probe_quality_report,
    )

    print(
        json.dumps(
            {
                "metrics_by_mode": metrics_by_mode,
                "distribution_by_mode": distribution_by_mode,
                "risk_by_mode": risk_by_mode,
                "probe_quality_report": probe_quality_report,
                "bm25_record_count": bm25_record_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
