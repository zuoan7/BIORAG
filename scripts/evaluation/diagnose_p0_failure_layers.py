#!/usr/bin/env python3
"""
Phase 1: P0 failure-layer diagnosis.
Calls RAG API for each calibrated P0 sample, extracts debug info,
and classifies the suspected failure layer.

Output:
  results/ragas/smoke100_20260430_113510/p0_failure_layer_diagnosis.json
  results/ragas/smoke100_20260430_113510/p0_failure_layer_diagnosis.md
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CALIBRATED_CSV_DIR = ROOT / "results/ragas/smoke100_20260430_113510"
P0_IDS = [
    # false_refusal_no_support
    "ent_065", "ent_071", "ent_100",
    # low_faithfulness_with_citations
    "ent_062", "ent_040", "ent_028", "ent_084", "ent_012",
    "ent_074", "ent_017", "ent_013", "ent_011", "ent_024",
    "ent_047", "ent_083",
]

# ── Keyword sets for false refusal diagnosis ─────────────────────
FALSE_REFUSAL_KEYWORDS: dict[str, list[str]] = {
    "ent_065": ["SpMAE", "spmae", "malate", "C4-dicarboxylate", "transporter",
                "malic enzyme", "maleate", "succinate", "fumarate"],
    "ent_071": ["positive selection", "negative selection", "Ni2+", "tetracycline",
                "resistance", "NeuAc", "riboswitch", "sialic acid", "sialyl",
                "antibiotic", "selectable marker"],
    "ent_100": ["circularly permuted", "cpFP", "fluorescent protein",
                "ligand", "conformational change", "chromophore", "GFP",
                "permutation", "biosensor"],
}

# ── Factoid entity patterns ──────────────────────────────────────
ENTITY_PATTERNS = re.compile(
    r"\b(?:[A-Z][a-z]{2,}(?:[A-Z][a-z]+)*\d*[A-Z]?)\b"  # gene/protein names
    r"|(?:CRISPR|Cas\d+|FadL|ABC|MFS|RND|MATE|SMR|TRAP|UGGT|HAC1)"
    r"|(?:pfkA|zwf|Fam20|Neu5Ac|GDP|UDP|CMP|ATP|NADH|NADPH)",
    re.IGNORECASE,
)
NUMERIC_PATTERNS = re.compile(
    r"\d+(?:\.\d+)?\s*(?:倍|fold|g/L|mg|mM|μM|nM|%|\s*h\b|\s*min\b)",
    re.IGNORECASE,
)

SECTION_FRAGMENT_SECTIONS = {
    "introduction", "results", "results and discussion", "discussion",
    "full text",
}
SECTION_GOOD_SECTIONS = {
    "abstract", "conclusion", "conclusions",
}


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_dataset() -> dict[str, dict[str, Any]]:
    path = ROOT / "data/eval/datasets/enterprise_ragas_eval_v1.json"
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {item["id"]: item for item in data}


def fetch_sample(base_url: str, dataset_item: dict[str, Any],
                 timeout: float) -> dict[str, Any]:
    """Call RAG API with include_debug=true for one sample."""
    payload = {
        "question": (dataset_item.get("question") or "").strip(),
        "session_id": f"diag_{dataset_item['id']}",
        "tenant_id": dataset_item.get("tenant_id", "default"),
        "doc_ids": dataset_item.get("doc_ids") or [],
        "sections": dataset_item.get("sections") or [],
        "source_files": dataset_item.get("source_files") or [],
        "include_debug": True,
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{base_url}/v1/ask", json=payload)
        resp.raise_for_status()
        return resp.json()


# ── Diagnostic logic ──────────────────────────────────────────────

def diagnose(item: dict[str, Any], api_resp: dict[str, Any],
             ragas_scores: dict[str, Any]) -> dict[str, Any]:
    sample_id = item["id"]
    question = item.get("question", "")
    answer = api_resp.get("answer", "")
    route = api_resp.get("route", "")
    debug = api_resp.get("debug") or {}
    gv2 = debug.get("generation_v2") or {}
    citations = api_resp.get("citations") or []
    support_pack = gv2.get("support_pack") or []
    candidates = gv2.get("candidates") or []
    answer_mode = gv2.get("answer_mode") or ""
    qwen_synthesis = gv2.get("qwen_synthesis") or {}
    has_qwen = bool(qwen_synthesis.get("used_qwen"))
    qwen_fallback = bool(qwen_synthesis.get("fallback_used"))
    refusal_reason = gv2.get("refuse_reason") or ""

    final_chunks_count = len(candidates)
    support_pack_count = len(support_pack)
    citation_count = len(citations)
    cited_sections = [c.get("section", "") for c in citations]

    # Extract candidate texts
    candidate_texts = []
    for c in candidates:
        text = (c.get("text") or c.get("chunk_text") or "")
        candidate_texts.append(text.lower())

    all_candidate_text = " ".join(candidate_texts)

    # RAGAS scores
    faith = ragas_scores.get("faithfulness")
    crec = ragas_scores.get("context_recall")
    cprec = ragas_scores.get("context_precision")

    # ── Diagnose ─────────────────────────────────────────────────
    failure_layer = "unknown"
    reason = ""

    if sample_id in FALSE_REFUSAL_KEYWORDS:
        keywords = FALSE_REFUSAL_KEYWORDS[sample_id]
        keyword_hits = [kw for kw in keywords if kw.lower() in all_candidate_text]
        if not keyword_hits:
            failure_layer = "retrieval_miss"
            reason = f"目标关键词未出现在final_chunks中: 搜索了{len(keywords)}个词, 0命中"
        elif support_pack_count == 0:
            failure_layer = "support_pack_miss"
            reason = f"final_chunks包含关键词({', '.join(keyword_hits[:3])})但support_pack=0"
        else:
            failure_layer = "citation_not_supporting_claim"
            reason = f"关键词在final_chunks中({', '.join(keyword_hits[:3])})但答案仍拒答, refusal_reason={refusal_reason}"
    elif route == "comparison":
        # Check branch coverage
        branches = gv2.get("covered_branches") or []
        missing = gv2.get("missing_branches") or []
        has_missing = len(missing) > 0
        answer_mentions_missing = any(
            kw in (answer or "").lower()
            for kw in ("未覆盖", "缺失", "不能比较", "只能进行有限比较",
                       "comparison_evidence_incomplete", "只能支持部分回答")
        )
        if has_missing and not answer_mentions_missing and citation_count > 0:
            failure_layer = "comparison_branch_miss"
            reason = (f"{len(missing)}个分支无证据但答案未明确说明"
                      f" (missing={missing[:2]}, covered={len(branches)})")
        elif faith is not None and faith < 0.5:
            section_lower = {s.lower() for s in cited_sections if s}
            has_fragment = bool(section_lower & SECTION_FRAGMENT_SECTIONS)
            if has_fragment:
                failure_layer = "summary_fragment_evidence"
                reason = f"comparison引用了非摘要碎片章节: {section_lower & SECTION_FRAGMENT_SECTIONS}"
            else:
                failure_layer = "citation_not_supporting_claim"
                reason = f"citation无法支撑comparison claims, faith={faith}"
        else:
            failure_layer = "judge_false_positive"
            reason = f"comparison faith={faith}可能judge误判"
    elif route == "summary":
        section_lower = {s.lower() for s in cited_sections if s}
        has_good = bool(section_lower & SECTION_GOOD_SECTIONS)
        has_fragment = bool(section_lower & SECTION_FRAGMENT_SECTIONS)
        # Count bibliography-like chunks
        bib_count = sum(
            1 for t in candidate_texts
            if re.search(r"https?://doi\.org", t)
            or t.count("http") > 10
            or re.search(r"\(\d{4}\)\.\s", t)
        )
        if faith is not None and faith < 0.3 and has_fragment and not has_good:
            failure_layer = "summary_fragment_evidence"
            reason = (f"引用碎片化正文章节{section_lower}, "
                      f"无Abstract/Conclusion, bibliography-like={bib_count}")
        elif has_qwen and faith is not None and faith < 0.5:
            failure_layer = "qwen_over_synthesis"
            reason = f"Qwen合成推断超出了support_pack范围, faith={faith}"
        else:
            failure_layer = "citation_not_supporting_claim"
            reason = f"summary faith={faith}, sections={section_lower}"
    elif route == "factoid":
        # Check entity/numeric overlap
        entities_in_q = set(re.findall(ENTITY_PATTERNS, question))
        entities_in_ans = set(re.findall(ENTITY_PATTERNS, answer))
        entities_in_cit = set(re.findall(ENTITY_PATTERNS, " ".join(
            c.get("quote", "") for c in citations
        )))
        numerics_in_ans = set(re.findall(NUMERIC_PATTERNS, answer))
        numerics_in_cit = set(re.findall(NUMERIC_PATTERNS, " ".join(
            c.get("quote", "") for c in citations
        )))

        entity_overlap = entities_in_ans & entities_in_cit if entities_in_ans else set()
        numeric_overlap = numerics_in_ans & numerics_in_cit if numerics_in_ans else set()
        entity_miss = entities_in_ans - entities_in_cit if entities_in_ans else set()
        numeric_miss = numerics_in_ans - numerics_in_cit if numerics_in_ans else set()

        if numeric_miss or entity_miss:
            failure_layer = "factoid_entity_or_numeric_mismatch"
            parts = []
            if numeric_miss:
                parts.append(f"数字不在citation中: {numeric_miss}")
            if entity_miss:
                parts.append(f"实体不在citation中: {list(entity_miss)[:5]}")
            reason = "; ".join(parts)
        elif citation_count <= 1 and faith is not None and faith < 0.5:
            failure_layer = "citation_not_supporting_claim"
            reason = f"仅有{citation_count}条citation但answer包含多项具体声明, faith={faith}"
        else:
            failure_layer = "qwen_over_synthesis"
            reason = f"factoid multi-claim answer中部分声明不被citation直接支撑, faith={faith}"
    else:
        # Route is "unknown" or unusual
        if citation_count == 0:
            failure_layer = "support_pack_miss"
            reason = f"zero citations, route={route}, mode={answer_mode}"
        elif faith is not None and faith < 0.5:
            failure_layer = "citation_not_supporting_claim"
            reason = f"low faithfulness on route={route}"
        else:
            failure_layer = "unknown"
            reason = f"unclassified failure on route={route}"

    return {
        "sample_id": sample_id,
        "route": route,
        "calibrated_issue_type": "",
        "question": question,
        "answer_mode": answer_mode or "unknown",
        "citation_count": citation_count,
        "doc_id_hit": ragas_scores.get("doc_id_hit", False),
        "section_norm_hit": ragas_scores.get("section_norm_hit", False),
        "context_recall": crec,
        "context_precision": cprec,
        "faithfulness": faith,
        "final_chunks_count": final_chunks_count,
        "support_pack_count": support_pack_count,
        "cited_chunks_count": citation_count,
        "has_qwen_synthesis": has_qwen,
        "qwen_fallback_used": qwen_fallback,
        "suspected_failure_layer": failure_layer,
        "diagnosis_reason": reason[:300],
        # Extra diagnostics
        "cited_sections": cited_sections,
        "refusal_reason": refusal_reason,
        "missing_branches": gv2.get("missing_branches") or [],
        "covered_branches": gv2.get("covered_branches") or [],
        "answer_preview": (answer or "")[:200],
    }


# ─── Main ─────────────────────────────────────────────────────────

def main() -> int:
    base_url = "http://127.0.0.1:9000"
    output_dir = CALIBRATED_CSV_DIR

    # Load dataset
    dataset_items = load_dataset()

    # Load existing ragas scores for metadata
    ragas_path = output_dir / "ragas_scores.jsonl"
    existing = load_jsonl(str(ragas_path))
    ragas_map: dict[str, dict[str, Any]] = {}
    for rec in existing:
        sid = rec.get("sample_id", "")
        scores = rec.get("ragas_scores") or {}
        ragas_map[sid] = {
            "doc_id_hit": rec.get("doc_id_hit"),
            "section_norm_hit": rec.get("section_norm_hit"),
            "faithfulness": scores.get("faithfulness"),
            "context_recall": scores.get("context_recall"),
            "context_precision": scores.get("context_precision"),
        }

    # Fetch debug data for each P0 sample
    diagnoses: list[dict[str, Any]] = []
    for idx, sid in enumerate(P0_IDS, start=1):
        ds_item = dataset_items.get(sid)
        if not ds_item:
            print(f"[diag] {sid}: NOT FOUND in dataset")
            continue

        print(f"[diag] {idx}/{len(P0_IDS)} Fetching debug for {sid}...")
        try:
            api_resp = fetch_sample(base_url, ds_item, timeout=300.0)
        except Exception as exc:
            print(f"[diag] {sid}: API ERROR: {exc}")
            diagnoses.append({
                "sample_id": sid,
                "route": "error",
                "suspected_failure_layer": "unknown",
                "diagnosis_reason": f"API call failed: {exc}",
            })
            continue

        rmap = ragas_map.get(sid, {})
        diag = diagnose(ds_item, api_resp, rmap)

        # Set calibrated issue type from known categories
        if sid in ("ent_065", "ent_071", "ent_100"):
            diag["calibrated_issue_type"] = "false_refusal_no_support"
        else:
            diag["calibrated_issue_type"] = "low_faithfulness_with_citations"

        print(f"  → {diag['suspected_failure_layer']}: {diag['diagnosis_reason'][:150]}")
        diagnoses.append(diag)

    # Write JSON
    json_path = output_dir / "p0_failure_layer_diagnosis.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(diagnoses, fh, ensure_ascii=False, indent=2)
    print(f"\n[diag] JSON → {json_path}")

    # Write MD report
    md = _build_md(diagnoses)
    md_path = output_dir / "p0_failure_layer_diagnosis.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"[diag] MD → {md_path}")

    # Summary
    layer_counts = Counter(d["suspected_failure_layer"] for d in diagnoses)
    print(f"\n[Failure Layer Summary]")
    for layer, count in layer_counts.most_common():
        print(f"  {layer}: {count}")

    return 0


def _build_md(diagnoses: list[dict[str, Any]]) -> str:
    lines = [
        "# P0 Failure-Layer Diagnosis",
        "",
        f"**Total P0 samples**: {len(diagnoses)}",
        "",
        "## Per-Sample Diagnosis",
        "",
    ]

    for d in diagnoses:
        lines += [
            f"### {d['sample_id']} — {d['suspected_failure_layer']}",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| route | `{d['route']}` |",
            f"| calibrated_issue_type | `{d['calibrated_issue_type']}` |",
            f"| answer_mode | `{d['answer_mode']}` |",
            f"| citation_count | {d['citation_count']} |",
            f"| doc_id_hit | {d['doc_id_hit']} |",
            f"| section_norm_hit | {d['section_norm_hit']} |",
            f"| context_recall | {d['context_recall']} |",
            f"| context_precision | {d['context_precision']} |",
            f"| faithfulness | {d['faithfulness']} |",
            f"| final_chunks_count | {d['final_chunks_count']} |",
            f"| support_pack_count | {d['support_pack_count']} |",
            f"| has_qwen_synthesis | {d['has_qwen_synthesis']} |",
            f"| qwen_fallback_used | {d['qwen_fallback_used']} |",
            f"| suspected_failure_layer | **{d['suspected_failure_layer']}** |",
            "",
            f"**Reason**: {d['diagnosis_reason']}",
            "",
            f"**Question**: {d['question']}",
            "",
            f"**Answer preview**: {d.get('answer_preview', '')}",
            "",
            "---",
            "",
        ]

    # Layer summary
    layer_counts = Counter(d["suspected_failure_layer"] for d in diagnoses)
    lines += [
        "## Failure Layer Summary",
        "",
        "| Layer | Count | Sample IDs |",
        "|-------|-------|------------|",
    ]
    for layer, count in layer_counts.most_common():
        ids = [d["sample_id"] for d in diagnoses if d["suspected_failure_layer"] == layer]
        lines.append(f"| {layer} | {count} | {', '.join(ids)} |")

    lines += [
        "",
        "## Diagnostic Rules Applied",
        "",
        "### false_refusal_no_support (3 samples)",
        "- 检查 final_chunks 中是否包含问题目标关键词",
        "- 关键词全部缺失 → `retrieval_miss`",
        "- 关键词在final_chunks但support_pack=0 → `support_pack_miss`",
        "",
        "### low_faithfulness_with_citations (12 samples)",
        "- comparison route: 检查分支缺失 + 答案披露情况",
        "- summary route: 检查引用章节是Abstract/Conclusion还是正文碎片",
        "- factoid route: 检查answer中的实体/数字是否直接出现在citation中",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
