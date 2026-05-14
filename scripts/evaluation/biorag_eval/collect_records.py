"""BIORAG Eval v1 — Collect EvalRecords from stable RAG pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .schemas import make_eval_record

BASE = Path(__file__).resolve().parent.parent.parent.parent
MAX_CONTEXTS = 3
MAX_CHARS_PER_CONTEXT = 1000
MAX_TOTAL_CONTEXT_CHARS = 3000


def collect_records(
    samples: list[dict[str, Any]],
    pipeline,
    smoke_results: dict[str, Any] | None = None,
    residual_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Run RAG pipeline on each sample and build EvalRecords.

    Args:
        samples: smoke200 dataset samples
        pipeline: SynBioRAGPipeline instance
        smoke_results: optional per-sample smoke results dict (sample_id → row)
        residual_ids: set of residual sample_ids
    """
    from src.synbio_rag.domain.schemas import QueryFilters

    smoke_by_id = smoke_results or {}
    residual_set = residual_ids or set()
    records = []
    stats = {"frozen": 0, "live": 0, "errors": 0}

    for s in samples:
        sid = s["sample_id"]
        try:
            resp = pipeline.answer(s["question"], filters=QueryFilters(tenant_id="default"))
        except Exception as e:
            stats["errors"] += 1
            records.append(make_eval_record(
                sample_id=sid, question=s["question"], answer=f"ERROR: {e}",
                is_negative=not s.get("expected_doc_ids"),
            ))
            continue

        qr = resp.debug.get("query_rewrite", {})
        if qr.get("rewrite_cache_source") == "frozen_eval_file":
            stats["frozen"] += 1
        else:
            stats["live"] += 1

        gv2 = resp.debug.get("generation_v2", {})
        support = _extract_support(gv2)

        ref = s.get("expected_answer") or ""
        meta = s.get("metadata")
        ak = ""
        if isinstance(meta, dict):
            ak = meta.get("answer_key") or ""
        ref_final = ref or ak

        citations = resp.citations or []
        cited_ids = list(set(c.doc_id for c in citations if c.doc_id))
        route_pred = resp.route.value if hasattr(resp.route, "value") else str(resp.route)

        # Smoke info
        smoke_row = smoke_by_id.get(sid, {})
        is_residual = sid in residual_set

        rec = make_eval_record(
            sample_id=sid,
            split="smoke150" if _in_smoke150(sid) else "added50",
            category=s.get("category", ""),
            expected_route=s.get("expected_route", ""),
            question=s["question"],
            answer=resp.answer or "",
            selected_support=support,
            cited_doc_ids=cited_ids,
            selected_support_doc_ids=[ss.get("doc_id", "") for ss in support],
            expected_doc_ids=s.get("expected_doc_ids") or [],
            expected_source_files=s.get("expected_source_files") or [],
            expected_behavior=_fmt_behavior(s.get("expected_behavior")),
            answer_key=ak,
            reference=ref_final,
            is_negative=not s.get("expected_doc_ids"),
            route_pred=route_pred,
            citation_count=len(citations),
            smoke_real_P0="yes" if is_residual else "no",
            smoke_failure_class=smoke_row.get("failure_class", ""),
            smoke_first_loss_stage=smoke_row.get("first_loss_stage", ""),
        )
        records.append(rec)

    return records


def _extract_support(gv2: dict) -> list[dict[str, str]]:
    """Extract slimmed selected_support from generation_v2 debug info."""
    candidates = gv2.get("candidates", [])
    cand_by_eid = {c.get("evidence_id", ""): c for c in candidates if c.get("evidence_id")}
    support_pack = gv2.get("support_pack", [])

    items = []
    for sp in support_pack:
        eid = sp.get("evidence_id", "")
        cand = cand_by_eid.get(eid)
        if cand and cand.get("text", "").strip():
            items.append({
                "support_id": eid,
                "doc_id": cand.get("doc_id", ""),
                "source_file": cand.get("source_file", ""),
                "text": cand["text"].strip(),
            })

    # Fallback to cited contexts
    if not items:
        for cit in gv2.get("citations", []):
            cand = cand_by_eid.get(cit.get("evidence_id", ""))
            text = cit.get("quote", "") or (cand.get("text", "") if cand else "")
            if text.strip():
                items.append({
                    "support_id": cit.get("evidence_id", ""),
                    "doc_id": cit.get("doc_id", cand.get("doc_id", "") if cand else ""),
                    "source_file": cit.get("source_file", cand.get("source_file", "") if cand else ""),
                    "text": text.strip(),
                })

    # Trim: max 3 items, 1000 chars each, 3000 total
    slimmed = []
    total = 0
    for item in items[:MAX_CONTEXTS]:
        text = item["text"]
        if len(text) > MAX_CHARS_PER_CONTEXT:
            text = text[:MAX_CHARS_PER_CONTEXT]
        remaining = MAX_TOTAL_CONTEXT_CHARS - total
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining]
        slimmed.append({**item, "text": text})
        total += len(text)
    return slimmed


def _in_smoke150(sid: str) -> bool:
    try:
        s150_ids = {json.loads(l)["sample_id"] for l in open(BASE / "data/eval/datasets/smoke150.jsonl")}
        return sid in s150_ids
    except Exception:
        return False


def _fmt_behavior(eb) -> str:
    if isinstance(eb, list):
        return ", ".join(str(x) for x in eb)
    return str(eb or "")
