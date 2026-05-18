from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain.schemas import RetrievedChunk

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.+\-/]*|[\u4e00-\u9fff]{1,4}", re.IGNORECASE)
_TABLE_QUERY_RE = re.compile(
    r"\b(table|row|column|caption|metric|value|yield|titer|residual|growth|chemostat)\b|表|行|列|指标|数值|产量|滴度",
    re.IGNORECASE,
)
_UNIT_OR_NOTE_QUERY_RE = re.compile(
    r"\b(note-aware|limitation|limitations|coordinate|coordinates|value-level|warning)\b|注释|限制",
    re.IGNORECASE,
)
_SOURCE_REFERENCE_QUERY_RE = re.compile(
    r"\b(gives? the reference|reference for|source for|provenance for|citation for)\b|参考|来源|出处",
    re.IGNORECASE,
)
_ROW_LOOKUP_QUERY_RE = re.compile(
    r"\b(row evidence|table row|which table row|find the row)\b|行证据|表格行",
    re.IGNORECASE,
)
_METRIC_LOOKUP_QUERY_RE = re.compile(
    r"\b(find metric evidence|metric evidence|structured table evidence lists|selected key values)\b|指标证据|数值证据",
    re.IGNORECASE,
)
_TABLE_LOOKUP_QUERY_RE = re.compile(
    r"\b(find the table|which table reports|table reports|table caption|caption)\b|表格|表",
    re.IGNORECASE,
)

_TYPE_AWARE_UNIT_BOOSTS: dict[str, dict[str, float]] = {
    "table_lookup": {"table_unit": 0.22, "row_unit": 0.03, "cell_group_unit": -0.02},
    "row_lookup": {"row_unit": 0.18, "cell_group_unit": 0.04, "table_unit": -0.04},
    "metric_lookup": {"cell_group_unit": 0.18, "row_unit": 0.04, "table_unit": -0.04},
    "source_or_reference_lookup": {
        "row_unit": 0.18,
        "cell_group_unit": 0.02,
        "table_unit": -0.04,
    },
    "unit_or_note_lookup": {"row_unit": 0.14, "cell_group_unit": 0.05, "table_unit": -0.02},
}
_TYPE_AWARE_UNIT_PRIORITY: dict[str, dict[str, int]] = {
    "table_lookup": {"table_unit": 0, "row_unit": 1, "cell_group_unit": 2},
    "row_lookup": {"row_unit": 0, "cell_group_unit": 1, "table_unit": 2},
    "metric_lookup": {"cell_group_unit": 0, "row_unit": 1, "table_unit": 2},
    "source_or_reference_lookup": {"row_unit": 0, "cell_group_unit": 1, "table_unit": 2},
    "unit_or_note_lookup": {"row_unit": 0, "cell_group_unit": 1, "table_unit": 2},
}


@dataclass
class TablePreviewCandidate:
    chunk: RetrievedChunk
    score: float
    rank: int = 0


class TablePreviewCandidateProvider:
    def __init__(self, units_path: str) -> None:
        self.units_path = units_path
        self._units: list[dict[str, Any]] | None = None
        self.last_debug: dict[str, Any] = {}

    def load_units(self) -> list[dict[str, Any]]:
        if self._units is not None:
            return self._units
        path = Path(self.units_path)
        units: list[dict[str, Any]] = []
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        value = json.loads(line)
                        if isinstance(value, dict):
                            units.append(value)
        self._units = units
        self.last_debug = {
            "units_path": str(path),
            "unit_count": len(units),
            "loaded": bool(units),
        }
        return units

    def search(self, question: str, *, top_k: int) -> list[TablePreviewCandidate]:
        units = self.load_units()
        scored: list[TablePreviewCandidate] = []
        for unit in units:
            score = _lexical_score(question, unit)
            if score <= 0:
                continue
            scored.append(
                TablePreviewCandidate(
                    chunk=adapt_table_preview_unit(unit, score=score),
                    score=score,
                )
            )
        scored.sort(key=lambda candidate: (-candidate.score, candidate.chunk.chunk_id))
        for rank, candidate in enumerate(scored[:top_k], start=1):
            candidate.rank = rank
            candidate.chunk.metadata["table_preview_rank"] = rank
        return scored[:top_k]


def apply_table_preview(
    *,
    question: str,
    retrieved: list[RetrievedChunk],
    config: Any,
    provider: TablePreviewCandidateProvider | None = None,
) -> tuple[list[RetrievedChunk], dict[str, Any]]:
    enabled = bool(getattr(config, "table_preview_enabled", False))
    debug: dict[str, Any] = {
        "enabled": enabled,
        "merge_enabled": bool(getattr(config, "table_preview_merge_enabled", False)),
        "merge_strategy": _table_preview_merge_strategy(config),
        "query_route": classify_table_preview_query(question),
        "mode": "disabled",
        "reason": "disabled",
        "input_count": len(retrieved),
        "output_count": len(retrieved),
        "candidate_count": 0,
        "merged_count": 0,
        "table_branch_executed": False,
        "table_candidates_in_rerank_input": False,
        "formal_citation_allowed": False,
        "candidate_chunk_ids": [],
    }
    if not enabled:
        return list(retrieved), debug

    units_path = str(
        getattr(
            config,
            "table_preview_units_path",
            "data/experiments/v7_phase7_table_index_unit_qa/phase7j_preview_eligible_units.jsonl",
        )
    )
    top_k = int(getattr(config, "table_preview_max_candidates", 20))
    provider = provider or TablePreviewCandidateProvider(units_path)
    candidates = provider.search(question, top_k=top_k)
    provider_debug = getattr(provider, "last_debug", {})
    debug.update(
        {
            "mode": "shadow",
            "reason": "shadow_debug_only",
            "units_path": units_path,
            "loaded_unit_count": provider_debug.get("unit_count", 0),
            "candidate_count": len(candidates),
            "table_branch_executed": True,
            "candidate_chunk_ids": [candidate.chunk.chunk_id for candidate in candidates],
            "candidate_table_index_unit_ids": [
                candidate.chunk.metadata.get("table_index_unit_id") for candidate in candidates
            ],
        }
    )

    if not bool(getattr(config, "table_preview_merge_enabled", False)):
        return list(retrieved), debug

    if not _is_table_like_query(question):
        debug.update(
            {
                "mode": "merge_blocked",
                "reason": "non_table_query_guard",
                "table_branch_executed": True,
            }
        )
        return list(retrieved), debug

    max_merge = int(getattr(config, "table_preview_merge_max_candidates", 5))
    min_score = float(getattr(config, "table_preview_min_score", 0.05))
    existing_ids = {chunk.chunk_id for chunk in retrieved}
    merge_strategy = _table_preview_merge_strategy(config)
    query_route = classify_table_preview_query(question)
    if merge_strategy == "type_aware_merge_v1":
        added = _select_type_aware_merge_candidates(
            question=question,
            candidates=candidates,
            existing_ids=existing_ids,
            max_merge=max_merge,
            min_score=min_score,
            config=config,
        )
    else:
        added = _select_baseline_merge_candidates(
            candidates=candidates,
            existing_ids=existing_ids,
            max_merge=max_merge,
            min_score=min_score,
        )
    merged = [*retrieved, *added]

    debug.update(
        {
            "mode": "merged_preview",
            "reason": "merge_enabled_for_table_like_query",
            "merge_strategy": merge_strategy,
            "query_route": query_route,
            "output_count": len(merged),
            "merged_count": len(added),
            "merged_chunk_ids": [chunk.chunk_id for chunk in added],
            "merged_table_index_unit_ids": [
                chunk.metadata.get("table_index_unit_id") for chunk in added
            ],
            "merged_unit_types": [chunk.metadata.get("table_unit_type") for chunk in added],
            "table_candidates_in_rerank_input": bool(added),
        }
    )
    return merged, debug


def adapt_table_preview_unit(unit: dict[str, Any], *, score: float = 0.0) -> RetrievedChunk:
    metadata = unit.get("metadata") if isinstance(unit.get("metadata"), dict) else {}
    provenance = unit.get("provenance") if isinstance(unit.get("provenance"), dict) else {}
    guardrail = unit.get("guardrail") if isinstance(unit.get("guardrail"), dict) else {}
    unit_id = str(unit.get("table_index_unit_id") or "")
    page = _parse_int(metadata.get("page"))
    row_label = metadata.get("row_label")
    table_unit_type = unit.get("unit_type")
    text = "\n".join(
        [
            "[TABLE INDEX UNIT]",
            f"[TABLE CAPTION] {_clean_caption(unit.get('caption'))}",
            f"unit_type: {table_unit_type or ''}",
            f"doc_id: {unit.get('doc_id') or ''}",
            f"table_id: {unit.get('table_id') or ''}",
            f"row_label: {'' if row_label is None else row_label}",
            f"[TABLE TEXT] {unit.get('content_text_for_embedding') or ''}",
            "limitations: preview_only; production_ready=false; "
            "value_bboxes_available=false; formal table citation disabled.",
        ]
    )
    adapted_metadata: dict[str, Any] = {
        "object_type": "table_index_unit",
        "table_preview": True,
        "table_preview_allow_formal_citation": False,
        "citation_formal_allowed": False,
        "table_index_unit_id": unit_id,
        "table_unit_type": table_unit_type,
        "seed_id": unit.get("seed_id"),
        "candidate_id": unit.get("candidate_id"),
        "doc_id": unit.get("doc_id"),
        "table_id": unit.get("table_id"),
        "caption": unit.get("caption"),
        "row_label": row_label,
        "header_path": metadata.get("header_path"),
        "row_values": metadata.get("row_values"),
        "cell_group_values": metadata.get("cell_group_values"),
        "retrieval_text": unit.get("content_text_for_embedding"),
        "source_csv_path": provenance.get("source_csv_path"),
        "source_pdf_crop_path": provenance.get("source_pdf_crop_path"),
        "source_markdown_path": provenance.get("source_markdown_path"),
        "source_span_granularity": provenance.get("source_span_granularity"),
        "value_bboxes_available": provenance.get("value_bboxes_available", False),
        "cell_bboxes_available": provenance.get("cell_bboxes_available"),
        "production_ready": guardrail.get("production_ready", False),
        "index_unit_status": guardrail.get("index_unit_status", "preview_only"),
        "binding_review_limitation": guardrail.get("binding_review_limitation"),
        "unit_or_note_ok": guardrail.get("unit_or_note_ok"),
        "reference_ok": guardrail.get("reference_ok"),
        "table_preview_score": round(score, 6),
    }
    return RetrievedChunk(
        chunk_id=f"table_preview::{unit_id}",
        doc_id=str(unit.get("doc_id") or ""),
        source_file="table_preview_debug_only",
        title=_clean_caption(unit.get("caption")),
        section=f"table_preview::{table_unit_type or 'table_index_unit'}",
        text=text,
        page_start=page,
        page_end=page,
        vector_score=round(score, 6),
        bm25_score=0.0,
        rerank_score=0.0,
        fusion_score=round(score, 6),
        metadata=adapted_metadata,
    )


def _lexical_score(question: str, unit: dict[str, Any]) -> float:
    query_tokens = _tokens(question)
    if not query_tokens:
        return 0.0
    text = " ".join(
        str(value or "")
        for value in (
            unit.get("doc_id"),
            unit.get("table_id"),
            unit.get("caption"),
            unit.get("content_text_for_embedding"),
        )
    )
    unit_tokens = _tokens(text)
    if not unit_tokens:
        return 0.0
    overlap = query_tokens & unit_tokens
    if not overlap:
        return 0.0
    return len(overlap) / max(len(query_tokens), 1)


def _tokens(value: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(value or "")}


def _is_table_like_query(question: str) -> bool:
    return bool(_TABLE_QUERY_RE.search(question or ""))


def classify_table_preview_query(question: str) -> str:
    text = question or ""
    if _UNIT_OR_NOTE_QUERY_RE.search(text):
        return "unit_or_note_lookup"
    if _SOURCE_REFERENCE_QUERY_RE.search(text):
        return "source_or_reference_lookup"
    if _ROW_LOOKUP_QUERY_RE.search(text):
        return "row_lookup"
    if _METRIC_LOOKUP_QUERY_RE.search(text):
        return "metric_lookup"
    if _TABLE_LOOKUP_QUERY_RE.search(text):
        return "table_lookup"
    if _is_table_like_query(text):
        return "table_lookup"
    return "non_table_query"


def _table_preview_merge_strategy(config: Any) -> str:
    strategy = str(getattr(config, "table_preview_merge_strategy", "") or "").strip().lower()
    if strategy == "type_aware_merge_v1":
        return strategy
    if bool(getattr(config, "table_preview_type_aware_merge_enabled", False)):
        return "type_aware_merge_v1"
    return "baseline_current"


def _select_baseline_merge_candidates(
    *,
    candidates: list[TablePreviewCandidate],
    existing_ids: set[str],
    max_merge: int,
    min_score: float,
) -> list[RetrievedChunk]:
    added: list[RetrievedChunk] = []
    seen_ids = set(existing_ids)
    for candidate in candidates:
        if len(added) >= max_merge:
            break
        if candidate.chunk.chunk_id in seen_ids:
            continue
        if candidate.score < min_score:
            continue
        candidate.chunk.metadata["table_preview_merge_strategy"] = "baseline_current"
        candidate.chunk.metadata["table_preview_merge_rank"] = len(added) + 1
        added.append(candidate.chunk)
        seen_ids.add(candidate.chunk.chunk_id)
    return added


def _select_type_aware_merge_candidates(
    *,
    question: str,
    candidates: list[TablePreviewCandidate],
    existing_ids: set[str],
    max_merge: int,
    min_score: float,
    config: Any,
) -> list[RetrievedChunk]:
    query_route = classify_table_preview_query(question)
    per_table_cap = max(
        1,
        int(getattr(config, "table_preview_type_aware_per_table_cap", max_merge)),
    )
    per_seed_cap = max(
        1,
        int(getattr(config, "table_preview_type_aware_per_seed_cap", max_merge)),
    )
    ranked: list[tuple[float, int, int, TablePreviewCandidate]] = []
    for lexical_rank, candidate in enumerate(candidates):
        if candidate.chunk.chunk_id in existing_ids:
            continue
        if candidate.score < min_score:
            continue
        unit_type = str(candidate.chunk.metadata.get("table_unit_type") or "")
        adjusted_score = candidate.score + _type_aware_unit_boost(query_route, unit_type)
        candidate.chunk.metadata["table_preview_raw_score"] = round(candidate.score, 6)
        candidate.chunk.metadata["table_preview_type_aware_score"] = round(adjusted_score, 6)
        candidate.chunk.metadata["table_preview_query_route"] = query_route
        candidate.chunk.metadata["table_preview_merge_strategy"] = "type_aware_merge_v1"
        ranked.append(
            (
                -adjusted_score,
                _type_aware_unit_priority(query_route, unit_type),
                lexical_rank,
                candidate,
            )
        )

    ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3].chunk.chunk_id))

    added: list[RetrievedChunk] = []
    seen_ids = set(existing_ids)
    table_counts: Counter[tuple[str, str]] = Counter()
    seed_counts: Counter[str] = Counter()
    seen_row_siblings: set[tuple[str, str, str]] = set()
    for _adjusted, _priority, _lexical_rank, candidate in ranked:
        if len(added) >= max_merge:
            break
        chunk = candidate.chunk
        if chunk.chunk_id in seen_ids:
            continue
        metadata = chunk.metadata
        table_key = (str(metadata.get("doc_id") or ""), str(metadata.get("table_id") or ""))
        seed_id = str(metadata.get("seed_id") or "")
        if table_counts[table_key] >= per_table_cap:
            continue
        if seed_id and seed_counts[seed_id] >= per_seed_cap:
            continue
        sibling_key = _row_cell_group_sibling_key(metadata)
        if sibling_key and sibling_key in seen_row_siblings:
            continue

        metadata["table_preview_merge_rank"] = len(added) + 1
        added.append(chunk)
        seen_ids.add(chunk.chunk_id)
        table_counts[table_key] += 1
        if seed_id:
            seed_counts[seed_id] += 1
        if sibling_key:
            seen_row_siblings.add(sibling_key)
    return added


def _type_aware_unit_boost(query_route: str, unit_type: str) -> float:
    return _TYPE_AWARE_UNIT_BOOSTS.get(query_route, {}).get(unit_type, -0.08)


def _type_aware_unit_priority(query_route: str, unit_type: str) -> int:
    return _TYPE_AWARE_UNIT_PRIORITY.get(query_route, {}).get(unit_type, 9)


def _row_cell_group_sibling_key(metadata: dict[str, Any]) -> tuple[str, str, str] | None:
    unit_type = metadata.get("table_unit_type")
    row_label = metadata.get("row_label")
    if unit_type not in {"row_unit", "cell_group_unit"} or not row_label:
        return None
    return (
        str(metadata.get("doc_id") or ""),
        str(metadata.get("table_id") or ""),
        str(row_label),
    )


def _parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_caption(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("[TABLE CAPTION]", "").strip()
