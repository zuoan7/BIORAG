#!/usr/bin/env python3
"""
Phase 2+3: Enhanced P0 diagnostics + minimal fix proposal.

Reads p0_failure_layer_diagnosis.json, fetches additional debug info,
adds structured diagnostic fields, and produces a fix proposal.

Output:
  results/ragas/smoke100_20260430_113510/p0_enhanced_diagnostics.json
  results/ragas/smoke100_20260430_113510/p0_fix_proposal.md
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DIAGNOSIS_PATH = ROOT / "results/ragas/smoke100_20260430_113510/p0_failure_layer_diagnosis.json"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_dataset() -> dict[str, dict[str, Any]]:
    path = ROOT / "data/eval/datasets/enterprise_ragas_eval_v1.json"
    return {item["id"]: item for item in load_json(str(path))}


_SECTION_TO_GROUP: dict[str, str] = {
    "Title": "TITLE", "Abstract": "ABSTRACT", "Introduction": "INTRO",
    "Background": "INTRO", "Results": "RESULT",
    "Results and Discussion": "RESULT", "Discussion": "DISCUSSION",
    "Conclusion": "CONCLUSION", "Conclusions": "CONCLUSION",
    "Methods": "METHOD", "Materials and Methods": "METHOD",
    "Experimental Section": "METHOD", "Experimental Procedures": "METHOD",
    "Full Text": "BODY_ANY",
}

_GOOD_SECTIONS = {"abstract", "conclusion", "conclusions"}
_FRAGMENT_SECTIONS = {"introduction", "results", "results and discussion",
                       "discussion", "full text", "methods",
                       "materials and methods", "experimental section"}

ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]{2,}(?:[A-Z][a-z]+)*\d*[A-Z]?)\b"
    r"|(?:CRISPR|Cas\d+|FadL|ABC|MFS|RND|MATE|SMR|TRAP|UGGT|HAC1"
    r"|pfkA|zwf|Fam20[A-C]?|Neu5Ac|GDP|UDP|CMP|ATP|NADH|NADPH"
    r"|SpMAE|TsaM|TsaT)",
    re.IGNORECASE,
)
NUMERIC_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:倍|fold|g/L|mg|mM|μM|nM|%|\s*h\b|\s*min\b|\s*℃|\s*°C)",
    re.IGNORECASE,
)
GENE_PATTERN = re.compile(
    r"\b[a-z]{3,4}[A-Z]\d*\b|"  # pfkA, zwf, etc.
    r"\b[A-Z][a-z]{2,}(?:[A-Z][a-z]+)+\b",  # Fam20C, etc.
)


def fetch_debug(base_url: str, dataset_item: dict[str, Any],
                timeout: float) -> dict[str, Any]:
    payload = {
        "question": dataset_item["question"],
        "session_id": f"enh_{dataset_item['id']}",
        "tenant_id": "default",
        "doc_ids": dataset_item.get("doc_ids") or [],
        "sections": dataset_item.get("sections") or [],
        "source_files": dataset_item.get("source_files") or [],
        "include_debug": True,
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{base_url}/v1/ask", json=payload)
        resp.raise_for_status()
        return resp.json()


def extract_entities(text: str) -> set[str]:
    """Extract gene/enzyme/protein/receptor entities from text."""
    found = set(ENTITY_PATTERN.findall(text))
    # Filter section names
    section_names = {"Abstract", "Introduction", "Results", "Discussion",
                     "Conclusion", "Methods", "Title", "Full Text",
                     "Results and Discussion", "Materials and Methods",
                     "Experimental Section", "Background", "Conclusions"}
    return {e for e in found if e not in section_names
            and len(e) >= 3 and not e.isdigit()}


def extract_numerics(text: str) -> set[str]:
    return set(NUMERIC_PATTERN.findall(text))


def enhance_diagnosis(item: dict[str, Any], api_resp: dict[str, Any],
                      ragas_meta: dict[str, Any]) -> dict[str, Any]:
    """Add structured diagnostic fields."""
    debug = api_resp.get("debug") or {}
    gv2 = debug.get("generation_v2") or {}
    citations = api_resp.get("citations") or []
    support_pack = gv2.get("support_pack") or []
    candidates = gv2.get("candidates") or []
    answer = api_resp.get("answer") or ""
    question = item.get("question", "")
    route = api_resp.get("route") or ""
    gv2_route = gv2.get("route") or route

    enhanced = dict(item)

    # ── 1. Support pack diagnostics ───────────────────────────────
    candidate_texts = [(c.get("text") or c.get("chunk_text") or "", c)
                       for c in candidates]

    rejection_reasons: list[dict[str, str]] = []
    if "support_selection_debug" in gv2:
        ssd = gv2["support_selection_debug"]
        excluded_list = ssd.get("excluded") or []
        for exc in excluded_list:
            rejection_reasons.append({
                "evidence_id": exc.get("evidence_id", ""),
                "reason": exc.get("reason", ""),
            })

    # Match question entities against candidates
    q_entities = extract_entities(question)
    q_numerics = extract_numerics(question)
    matched_entities: set[str] = set()
    for text, c in candidate_texts:
        ents = extract_entities(text)
        matched_entities.update(q_entities & ents)

    enhanced["support_pack_diagnostics"] = {
        "support_pack_count": len(support_pack),
        "rejected_candidate_count": len(rejection_reasons),
        "rejection_reasons": rejection_reasons[:10],
        "matched_question_entities": sorted(matched_entities)[:15],
        "missing_question_entities": sorted(q_entities - matched_entities)[:15],
    }

    # ── 2. Citation diagnostics ──────────────────────────────────
    cited_chunk_ids = [c.get("chunk_id", "") for c in citations]
    cited_sections = [c.get("section", "") for c in citations]
    cited_docs = [c.get("doc_id", "") for c in citations]

    # Simple claim-to-citation placeholder
    claim_map: list[dict[str, Any]] = []
    answer_sentences = re.split(r"[。；\n]", answer)
    for sent in answer_sentences[:10]:
        sent = sent.strip()
        if len(sent) < 10:
            continue
        entities_in_sent = extract_entities(sent)
        supporting_citations: list[int] = []
        for ci, c in enumerate(citations):
            quote = (c.get("quote") or "").lower()
            for ent in entities_in_sent:
                if ent.lower() in quote:
                    supporting_citations.append(ci)
                    break
        claim_map.append({
            "sentence": sent[:120],
            "entities": sorted(entities_in_sent)[:8],
            "likely_supporting_citation_idx": list(set(supporting_citations))[:3],
        })

    enhanced["citation_diagnostics"] = {
        "citation_count": len(citations),
        "cited_chunk_ids": cited_chunk_ids,
        "cited_chunk_sections": cited_sections,
        "cited_chunk_docs": cited_docs,
        "claim_to_citation_map": claim_map,
    }

    # ── 3. Route-specific diagnostics ────────────────────────────
    enhanced["route_diagnostics"] = {}

    if gv2_route == "comparison":
        branches = gv2.get("covered_branches") or []
        missing = gv2.get("missing_branches") or []
        branch_details: list[dict[str, Any]] = []

        all_branch_labels = set()
        for b in branches:
            all_branch_labels.add(b if isinstance(b, str) else b.get("label", str(b)))
        for b in missing:
            all_branch_labels.add(b if isinstance(b, str) else b.get("label", str(b)))

        for label in all_branch_labels:
            branch_citations = [
                ci for ci, c in enumerate(citations)
                if label.lower() in (c.get("quote") or "").lower()
            ]
            branch_details.append({
                "branch_name": label,
                "is_covered": label in (branches if isinstance(branches[0], str)
                                        else [br.get("label", "") for br in branches])
                               if branches else False,
                "branch_citation_count": len(branch_citations),
            })

        enhanced["route_diagnostics"]["comparison"] = {
            "total_branches": len(all_branch_labels),
            "covered_branches": len(branches),
            "missing_branches": missing if isinstance(missing, list) else [],
            "missing_count": len(missing),
            "branch_details": branch_details,
        }

    elif gv2_route == "summary":
        section_dist: dict[str, int] = Counter()
        abstract_conclusion_count = 0
        fragment_body_count = 0
        bibliography_count = 0

        for c in candidates:
            section = (c.get("section") or "").lower()
            section_dist[section] += 1
            if section in _GOOD_SECTIONS:
                abstract_conclusion_count += 1
            elif section in _FRAGMENT_SECTIONS:
                fragment_body_count += 1

            text = (c.get("text") or c.get("chunk_text") or "").lower()
            if (re.search(r"https?://doi\.org", text)
                    or text.count("http") > 8
                    or re.search(r"et\s+al\.\s+\d{4}", text)):
                bibliography_count += 1

        enhanced["route_diagnostics"]["summary"] = {
            "section_distribution": dict(section_dist),
            "abstract_or_conclusion_support_count": abstract_conclusion_count,
            "fragmentary_body_support_count": fragment_body_count,
            "bibliography_like_chunk_count": bibliography_count,
        }

    elif gv2_route == "factoid":
        # Required entities/numerics
        required_entities = extract_entities(question)
        required_numerics = extract_numerics(question)
        answer_entities = extract_entities(answer)
        answer_numerics = extract_numerics(answer)
        citation_text = " ".join(c.get("quote", "") for c in citations)
        citation_entities = extract_entities(citation_text)
        citation_numerics = extract_numerics(citation_text)

        enhanced["route_diagnostics"]["factoid"] = {
            "required_entity_terms": sorted(required_entities)[:10],
            "required_numeric_terms": sorted(required_numerics)[:10],
            "answer_entity_terms": sorted(answer_entities)[:15],
            "answer_numeric_terms": sorted(answer_numerics)[:10],
            "citation_entity_overlap": sorted(answer_entities & citation_entities)[:10],
            "citation_entity_miss": sorted(answer_entities - citation_entities)[:10],
            "citation_numeric_overlap": sorted(answer_numerics & citation_numerics)[:10],
            "citation_numeric_miss": sorted(answer_numerics - citation_numerics)[:10],
        }

    return enhanced


def build_fix_proposal(enhanced_diags: list[dict[str, Any]]) -> str:
    """Phase 3: minimal fix proposal based on diagnosis."""
    layer_counts = Counter(d["suspected_failure_layer"] for d in enhanced_diags)
    total = len(enhanced_diags)

    lines = [
        "# P0 Fix Proposal (Phase 3)",
        "",
        f"**Based on**: {total} calibrated P0 samples",
        "",
        "## Failure Layer Distribution",
        "",
        "| Layer | Count | % |",
        "|-------|-------|---|",
    ]
    for layer, count in layer_counts.most_common():
        lines.append(f"| {layer} | {count} | {count/total*100:.0f}% |")

    # Group samples by layer
    layer_samples: dict[str, list[str]] = defaultdict(list)
    for d in enhanced_diags:
        layer_samples[d["suspected_failure_layer"]].append(d["sample_id"])

    lines += [
        "",
        "## Recommended Fixes (max 3)",
        "",
    ]

    # Fix A: Summary fragment evidence (6 samples)
    if layer_counts.get("summary_fragment_evidence", 0) >= 3:
        ids = layer_samples["summary_fragment_evidence"]
        lines += [
            "### Fix A: Summary support selection — prefer Abstract/Conclusion",
            "",
            f"**Coverage**: {len(ids)} samples — {', '.join(ids)}",
            "",
            "**Problem**: Summary route cites fragmented body sections (Results, Introduction, "
            "Discussion) instead of Abstract/Conclusion/Results summary. This causes Qwen to "
            "synthesize from partial evidence, producing incomplete or misleading summaries.",
            "",
            "**Proposed fix**:",
            "- In summary support selection, boost chunks from Abstract, Conclusion, Conclusions sections",
            "- Deprioritize pure body-text fragments when better sections exist",
            "- If only fragmentary evidence exists, answer_mode must be partial_summary with explicit caveat",
            "",
            "**Expected impact**: faithfulness improvement on affected samples",
            "**Risk**: May reduce answer completeness if Abstract is too terse",
            "**Recommendation**: **Implement in this round** (covers 6/15 = 40% of P0)",
            "",
        ]

    # Fix B: Support pack miss (3 samples)
    if layer_counts.get("support_pack_miss", 0) >= 3:
        ids = layer_samples["support_pack_miss"]
        lines += [
            "### Fix B: no_support_pack limited fallback",
            "",
            f"**Coverage**: {len(ids)} samples — {', '.join(ids)}",
            "",
            "**Problem**: Final chunks contain entities matching the question, but "
            "support_pack is empty. The support selector rejects all candidates "
            "due to strict thresholds, causing false refusals.",
            "",
            "**Proposed fix**:",
            "- When support_pack=0 but final_chunks contain question entity hits, "
              "create a limited_support_pack from the top-N entity-matching chunks",
            "- Flag as limited_support_pack_used=true in debug",
            "- Apply stricter faithfulness checks on the limited answer",
            "",
            "**Expected impact**: 3 false refusals converted to partial answers",
            "**Risk**: May introduce low-quality support; guard with entity-match requirement",
            "**Recommendation**: **Implement in this round** (covers 3/15 = 20% of P0)",
            "",
        ]

    # Fix C: Factoid entity/numeric mismatch (2 samples)
    if layer_counts.get("factoid_entity_or_numeric_mismatch", 0) >= 2:
        ids = layer_samples["factoid_entity_or_numeric_mismatch"]
        lines += [
            "### Fix C: Factoid entity/numeric citation validation",
            "",
            f"**Coverage**: {len(ids)} samples — {', '.join(ids)}",
            "",
            "**Problem**: Factoid answers contain entities or numbers not directly "
            "present in cited chunks. Qwen may fabricate or infer values.",
            "",
            "**Proposed fix**:",
            "- Post-generation check: verify key entities/numbers in answer appear in "
              "cited chunk quotes (string-level)",
            "- If mismatch detected, remove the unsupported sentence or mark as partial",
            "- Conservative: only flag, don't delete, until human calibration",
            "",
            "**Expected impact**: Reduced hallucination on factoid questions",
            "**Risk**: String matching may miss synonym variations (e.g. Neu5Ac vs sialic acid)",
            "**Recommendation**: **Defer** — only 2/15 P0 (13%), needs terminology mapping first",
            "",
        ]

    # Fix D: citation_not_supporting_claim (4 samples) — not directly actionable
    if layer_counts.get("citation_not_supporting_claim", 0) >= 3:
        ids = layer_samples["citation_not_supporting_claim"]
        lines += [
            "### Fix D: Citation-not-supporting-claim — indirect",
            "",
            f"**Coverage**: {len(ids)} samples — {', '.join(ids)}",
            "",
            "**Problem**: These samples have citations but RAGAS judge scores them low. "
            "Root cause overlaps with Fix A (summary_fragment_evidence) — once "
            "better sections are selected, citation quality should improve.",
            "",
            "**Recommendation**: **Defer** — address via Fix A first, then re-evaluate",
            "",
        ]

    lines += [
        "## Summary",
        "",
        "| Priority | Fix | Coverage | Action |",
        "|----------|-----|----------|--------|",
    ]
    if layer_counts.get("summary_fragment_evidence", 0) >= 3:
        lines.append(f"| **1** | Fix A: Summary Abstract/Conclusion boost | {layer_counts['summary_fragment_evidence']}/15 | ✅ Implement |")
    if layer_counts.get("support_pack_miss", 0) >= 3:
        lines.append(f"| **2** | Fix B: limited_support_pack fallback | {layer_counts['support_pack_miss']}/15 | ✅ Implement |")
    if layer_counts.get("factoid_entity_or_numeric_mismatch", 0) >= 2:
        lines.append(f"| 3 | Fix C: factoid entity validation | {layer_counts['factoid_entity_or_numeric_mismatch']}/15 | ⏭️ Defer |")

    lines += [
        "",
        f"**Combined coverage**: Fix A+B = {layer_counts.get('summary_fragment_evidence', 0) + layer_counts.get('support_pack_miss', 0)}/15 = "
        f"{(layer_counts.get('summary_fragment_evidence', 0) + layer_counts.get('support_pack_miss', 0)) / total * 100:.0f}% of P0 samples",
    ]

    return "\n".join(lines)


def main() -> int:
    base_url = "http://127.0.0.1:9000"
    output_dir = ROOT / "results/ragas/smoke100_20260430_113510"

    diags = load_json(str(DIAGNOSIS_PATH))
    dataset_items = load_dataset()

    # Load ragas metadata (JSONL)
    ragas_map: dict[str, dict[str, Any]] = {}
    ragas_jsonl_path = output_dir / "ragas_scores.jsonl"
    with open(ragas_jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("sample_id", "")
            if sid:
                ragas_map[sid] = rec

    enhanced: list[dict[str, Any]] = []
    for idx, diag in enumerate(diags, start=1):
        sid = diag["sample_id"]
        ds_item = dataset_items.get(sid)
        if not ds_item:
            enhanced.append(diag)
            continue

        print(f"[enhance] {idx}/{len(diags)} {sid}...")
        try:
            api_resp = fetch_debug(base_url, ds_item, timeout=300.0)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            enhanced.append(diag)
            continue

        enh = enhance_diagnosis(diag, api_resp, ragas_map.get(sid, {}))
        print(f"  → sp={enh['support_pack_diagnostics']['support_pack_count']} "
              f"cit={enh['citation_diagnostics']['citation_count']} "
              f"rej={enh['support_pack_diagnostics']['rejected_candidate_count']}")
        enhanced.append(enh)

    # Write enhanced JSON
    json_path = output_dir / "p0_enhanced_diagnostics.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(enhanced, fh, ensure_ascii=False, indent=2)
    print(f"\n[enhance] JSON → {json_path}")

    # Write fix proposal
    fix_proposal = build_fix_proposal(enhanced)
    md_path = output_dir / "p0_fix_proposal.md"
    md_path.write_text(fix_proposal, encoding="utf-8")
    print(f"[enhance] Fix proposal → {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
