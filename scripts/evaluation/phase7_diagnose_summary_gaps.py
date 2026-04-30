#!/usr/bin/env python3
"""
Phase 7A: Deep diagnosis of summary retrieval gaps for 10 summary_fragment_evidence P0 samples.
Checks each stage: KB → retrieved → reranked → final_chunks → support_pack → citation → answer.
"""
from __future__ import annotations
import json, csv, sys, re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import httpx

ROOT = Path(__file__).resolve().parents[2]
P5_DIR = ROOT / "results/ragas/smoke100_20260430_153147"

SUMMARY_P0_IDS = [
    "ent_013", "ent_024", "ent_040", "ent_062", "ent_084",
    "ent_015", "ent_032", "ent_036", "ent_042", "ent_051",
]

GOOD_SECTIONS = {"abstract", "conclusion", "conclusions",
                 "results and discussion", "discussion and results"}
SUMMARY_LIKE_PATTERNS = re.compile(
    r"abstract|conclusion|summary|outlook|perspective|overview",
    re.IGNORECASE,
)

def load_jsonl(path: str) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            recs.append(json.loads(line))
    return recs

def load_ds():
    with open(ROOT / "data/eval/datasets/enterprise_ragas_eval_v1.json") as f:
        return {i["id"]: i for i in json.load(f)}

def fetch(sid, ds, base_url="http://127.0.0.1:9000"):
    item = ds[sid]
    payload = {"question": item["question"], "session_id": f"p7a_{sid}", "tenant_id": "default",
               "doc_ids": item.get("doc_ids") or [], "sections": item.get("sections") or [],
               "source_files": item.get("source_files") or [], "include_debug": True}
    with httpx.Client(timeout=300) as c:
        resp = c.post(f"{base_url}/v1/ask", json=payload)
        resp.raise_for_status()
        return resp.json()

def main():
    ds = load_ds()
    p5_scores = {s["sample_id"]: s for s in load_jsonl(str(P5_DIR / "ragas_scores.jsonl"))}

    diagnoses = []
    for idx, sid in enumerate(SUMMARY_P0_IDS, 1):
        print(f"[{idx}/{len(SUMMARY_P0_IDS)}] {sid}...")
        resp = fetch(sid, ds)
        gv2 = (resp.get("debug", {}).get("generation_v2") or {})
        candidates = gv2.get("candidates") or []
        support_pack = gv2.get("support_pack") or []
        citations = resp.get("citations") or []
        ledger = gv2.get("retrieval_ledger") or {}
        p5 = p5_scores.get(sid, {})
        sc = p5.get("ragas_scores") or {}

        # Doc tracking
        expected_docs = ds[sid].get("expected_doc_ids", [])
        retrieved_docs = sorted(set(c.get("doc_id","") for c in candidates))
        sp_docs = sorted(set(s.get("doc_id","") for s in support_pack))
        cit_docs = sorted(set(c.get("doc_id","") for c in citations))

        # Doc hit stage
        if set(expected_docs) & set(cit_docs):
            doc_stage = "citation"
        elif set(expected_docs) & set(sp_docs):
            doc_stage = "support_pack"
        elif set(expected_docs) & set(retrieved_docs):
            doc_stage = "final_chunks"
        elif retrieved_docs:
            doc_stage = "retrieved"
        else:
            doc_stage = "none"

        # Section tracking
        candidate_sections = [c.get("section","") for c in candidates]
        sp_sections = [s.get("section","") for s in support_pack]
        cit_sections = [c.get("section","") for c in citations]

        has_abs_in_candidates = any(s.lower() == "abstract" for s in candidate_sections)
        has_conc_in_candidates = any("conclusion" in s.lower() for s in candidate_sections)
        has_abs_in_sp = any(s.lower() == "abstract" for s in sp_sections)
        has_conc_in_sp = any("conclusion" in s.lower() for s in sp_sections)
        has_abs_in_cit = any(s.lower() == "abstract" for s in cit_sections)
        has_conc_in_cit = any("conclusion" in s.lower() for s in cit_sections)

        # Check if any candidate title/section suggests summary-like content
        summary_like = []
        for c in candidates:
            title = c.get("title","")
            section = c.get("section","")
            if SUMMARY_LIKE_PATTERNS.search(f"{title} {section}"):
                summary_like.append(f"{c.get('doc_id')}|{section}")

        # Classify failure subtype
        if doc_stage in ("none",):
            subtype = "doc_retrieval_miss"
        elif not has_abs_in_candidates and not has_conc_in_candidates:
            subtype = "summary_section_not_retrieved"
        elif has_abs_in_candidates and not has_abs_in_sp:
            subtype = "summary_section_retrieved_not_selected"
        elif has_abs_in_sp and not has_abs_in_cit:
            subtype = "summary_section_selected_not_cited"
        elif has_abs_in_cit:
            subtype = "summary_section_selected_but_answer_fragmentary"
        else:
            subtype = "unknown"

        # Also check if summary-like chunks exist in KB (via whether ANY doc has abstract/conclusion)
        # This is a heuristic: if any candidate from the same doc has abstract, it exists in KB

        diag = {
            "sample_id": sid,
            "question": ds[sid]["question"][:120],
            "route": resp.get("route"),
            "expected_doc_ids": expected_docs,
            "retrieved_doc_ids": retrieved_docs,
            "final_chunk_doc_ids": retrieved_docs,
            "support_pack_doc_ids": sp_docs,
            "citation_doc_ids": cit_docs,
            "doc_level_hit_stage": doc_stage,
            "final_chunks_sections": candidate_sections[:8],
            "support_pack_sections": sp_sections[:5],
            "citation_sections": cit_sections[:5],
            "has_abstract_in_candidates": has_abs_in_candidates,
            "has_conclusion_in_candidates": has_conc_in_candidates,
            "abstract_in_support_pack": has_abs_in_sp,
            "conclusion_in_support_pack": has_conc_in_sp,
            "abstract_in_citation": has_abs_in_cit,
            "conclusion_in_citation": has_conc_in_cit,
            "summary_like_chunk_available": bool(summary_like),
            "summary_like_chunks": summary_like[:5],
            "final_chunks_count": len(candidates),
            "support_pack_count": len(support_pack),
            "citation_count": len(citations),
            "current_failure_subtype": subtype,
            "faithfulness": sc.get("faithfulness"),
            "context_recall": sc.get("context_recall"),
        }
        print(f"  doc={doc_stage} subtype={subtype} "
              f"abs_cand={has_abs_in_candidates} abs_sp={has_abs_in_sp} "
              f"conc_cand={has_conc_in_candidates} summary_like={len(summary_like)}")
        diagnoses.append(diag)

    # Write outputs
    out_dir = P5_DIR
    _write_csv(diagnoses, out_dir / "phase7_summary_retrieval_gap_diagnosis.csv")
    _write_md(diagnoses, out_dir / "phase7_summary_retrieval_gap_diagnosis.md")
    _write_json(diagnoses, out_dir / "phase7_summary_retrieval_gap_diagnosis.json")

    # Summary stats
    subtypes = Counter(d["current_failure_subtype"] for d in diagnoses)
    print(f"\n=== Failure Subtypes ===")
    for k, v in subtypes.most_common():
        ids = [d["sample_id"] for d in diagnoses if d["current_failure_subtype"] == k]
        print(f"  {k}: {v} — {ids}")

    not_retrieved = subtypes.get("summary_section_not_retrieved", 0)
    not_selected = subtypes.get("summary_section_retrieved_not_selected", 0)
    total_gap = not_retrieved + not_selected
    print(f"\nPhase 7B gate: not_retrieved={not_retrieved}, not_selected={not_selected}, total_gap={total_gap}")
    print(f"Gate condition (total_gap >= 3): {'✅ PASS' if total_gap >= 3 else '❌ FAIL'}")


def _write_csv(diags, path):
    if not diags: return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(diags[0].keys()))
        w.writeheader()
        w.writerows(diags)

def _write_md(diags, path):
    lines = ["# Phase 7A: Summary Retrieval Gap Diagnosis", "",
             f"**Samples**: {len(diags)}", "",
             "| ID | Doc Stage | Abs in Cand | Abs in SP | Abs in Cit | Conc in Cand | Summary-like | Subtype |",
             "|----|-----------|-------------|-----------|------------|--------------|--------------|---------|"]
    for d in diags:
        lines.append(f"| {d['sample_id']} | {d['doc_level_hit_stage']} | "
                     f"{d['has_abstract_in_candidates']} | {d['abstract_in_support_pack']} | "
                     f"{d['abstract_in_citation']} | {d['has_conclusion_in_candidates']} | "
                     f"{d['summary_like_chunk_available']} | **{d['current_failure_subtype']}** |")

    subtypes = Counter(d["current_failure_subtype"] for d in diags)
    lines += ["", "## Subtype Distribution", ""]
    for k, v in subtypes.most_common():
        ids = [d["sample_id"] for d in diags if d["current_failure_subtype"] == k]
        lines.append(f"- **{k}**: {v} — {', '.join(f'`{i}`' for i in ids)}")

    lines += ["", "## Per-Sample Detail", ""]
    for d in diags:
        lines += [
            f"### {d['sample_id']} — {d['current_failure_subtype']}",
            f"- Doc stage: `{d['doc_level_hit_stage']}`",
            f"- Expected docs: {d['expected_doc_ids']}",
            f"- Retrieved docs: {d['retrieved_doc_ids'][:5]}",
            f"- Final chunk sections: {d['final_chunks_sections'][:5]}",
            f"- Support sections: {d['support_pack_sections'][:5]}",
            f"- Citation sections: {d['citation_sections'][:5]}",
            f"- Summary-like available: {d['summary_like_chunk_available']}",
            f"- Summary-like chunks: {d['summary_like_chunks'][:5]}",
            f"- Question: {d['question']}", "",
        ]

    not_retrieved = subtypes.get("summary_section_not_retrieved", 0)
    not_selected = subtypes.get("summary_section_retrieved_not_selected", 0)
    lines += ["## Phase 7B Gate", "",
              f"- summary_section_not_retrieved: {not_retrieved}",
              f"- summary_section_retrieved_not_selected: {not_selected}",
              f"- **Total gap**: {not_retrieved + not_selected}",
              f"- **Gate condition (>=3)**: {'✅ PASS — proceed to Phase 7C' if not_retrieved + not_selected >= 3 else '❌ FAIL — stop, only output diagnosis'}"]

    path.write_text("\n".join(lines), encoding="utf-8")

def _write_json(diags, path):
    path.write_text(json.dumps(diags, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
