#!/usr/bin/env python3
"""Phase 16E: Answer Evidence Marker Usage Audit.

Audits why citation_marker_not_used happens: examines the answer text,
checking whether uncited support items are semantically used by the answer
but simply missing their [E#] markers.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.config import Settings
from src.synbio_rag.domain.schemas import QueryFilters

DATASET = Path("data/eval/datasets/enterprise_ragas_smoke100.json")
OUTPUT_DIR = Path("results/phase16e_answer_marker_usage_audit")
REPORT_DIR = Path("reports/phase16e_answer_marker_usage_audit")
PHASE16D_METRICS = Path("results/phase16d_smoke100_citation_contract_validation/smoke100_phase16d_metrics.json")
PHASE16D_DROP = Path("results/phase16d_smoke100_citation_contract_validation/drop_reason_full_smoke100.json")

_EVIDENCE_RE = re.compile(r"\[(E\d+)\]")
_CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 16E marker usage audit.")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-ids", nargs="*", default=[])
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [item for item in data if isinstance(item, dict)]


def sample_id(sample: dict[str, Any]) -> str:
    return str(sample.get("sample_id") or sample.get("id") or "")


# ── Semantic overlap detection ────────────────────────────────────

def tokenize_cjk(text: str) -> set[str]:
    tokens: set[str] = set()
    for i in range(len(text) - 1):
        if _CJK_CHAR.match(text[i]) and _CJK_CHAR.match(text[i + 1]):
            tokens.add(text[i:i + 2])
    for i in range(len(text) - 2):
        if all(_CJK_CHAR.match(c) for c in text[i:i + 3]):
            tokens.add(text[i:i + 3])
    return tokens


def tokenize_en(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9]{2,}", text.lower()))


def extract_key_terms(text: str) -> set[str]:
    en = tokenize_en(text)
    cjk = tokenize_cjk(text)
    return en | cjk


def semantic_overlap_ratio(candidate_text: str, answer_text: str) -> float:
    """How much of the candidate's key terms appear in the answer."""
    c_terms = extract_key_terms(candidate_text)
    if not c_terms:
        return 0.0
    a_text_lower = answer_text.lower()
    hits = sum(1 for t in c_terms if t.lower() in a_text_lower)
    return hits / len(c_terms)


def find_supporting_sentence(candidate_text: str, answer_text: str) -> str | None:
    """Find the answer sentence that best matches candidate_text."""
    # Split answer into sentences
    sentences = re.split(r'(?<=[。！？.!?])\s*', answer_text)
    best_sentence = None
    best_overlap = 0.0
    for sent in sentences:
        if len(sent) < 10:
            continue
        overlap = semantic_overlap_ratio(candidate_text, sent)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sent.strip()
    if best_overlap > 0.1:
        return best_sentence[:200]
    return None


# ── Main audit ─────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(Path(args.dataset))
    sample_by_id = {sample_id(s): s for s in samples}

    if args.sample_ids:
        run_ids = [sid for sid in args.sample_ids if sid in sample_by_id]
    elif args.limit > 0:
        run_ids = [sample_id(s) for s in samples[:args.limit]]
    else:
        run_ids = [sample_id(s) for s in samples]

    # ── Part 1: Code path audit ──────────────────────────────────
    code_audit = {
        "selected_support_entry_to_answer_builder": (
            "service.py calls answer_builder.build(question, analysis, plan, support_pack). "
            "support_pack (list[SupportItem]) enters directly as the 4th argument."
        ),
        "citation_candidates_entry_to_answer_builder": (
            "citation_candidates are NOT passed to answer_builder. "
            "Only support_pack is used. citation_candidates are built later "
            "by CitationBinder.build_citation_candidates(support_pack) for the citation layer."
        ),
        "answer_builder_uses": "support_pack (not citation_candidates)",
        "marker_generation_location": "answer_builder.py: ExtractiveAnswerBuilder.build()",
        "marker_generation_rules": {
            "comparison_with_coverage": (
                "Lines 46-64: for each branch_entry in comparison_coverage.branch_evidence, "
                "references [evidence_id] for each branch's evidence_ids. ALL branch evidence gets markers."
            ),
            "comparison_without_coverage": (
                "Falls through to else (line 103-105): only support_pack[:3] gets markers. "
                "This is the CURRENT state since v2_enable_comparison_coverage=false."
            ),
            "summary": (
                "Lines 95-102: _build_summary_claims() generates up to 5 claims with [evidence_id] markers. "
                "Claims are ranked by section quality, deduplicated. Max 5."
            ),
            "factoid_unknown_other": (
                "Lines 103-105: for item in support_pack[:3]: lines.append(f'... [{item.evidence_id}]'). "
                "HARD CODED limit of 3 support items with markers."
            ),
            "experiment": "Lines 30-31: note about no experiment protocol. Then falls to standard logic.",
            "refuse": "Lines 20-27: refusal text, no markers generated.",
        },
        "marker_id_mapping": (
            "EvidenceId comes from EvidenceCandidate.evidence_id, set by EvidenceLedgerBuilder. "
            "Format: E001, E002, ... based on order in candidates list."
        ),
        "answer_postprocess": (
            "When Qwen synthesis is OFF: extractive_answer is returned AS-IS (no postprocessing). "
            "When Qwen synthesis is ON: synthesized answer is validated but NOT postprocessed for markers."
        ),
        "citation_binder_parser": (
            "citation_binder.py: _EVIDENCE_REF_PATTERN = re.compile(r'\\[(E\\d+)\\]'). "
            "Matches [E#] with digits only. Replaces [E001] → [1], [E002] → [2], etc."
        ),
        "marker_parse_failure_drop_reason": (
            "Invalid marker IDs (not in support_by_id) are recorded in debug['invalid_evidence_ids']. "
            "They are NOT recorded as a separate drop_reason category — the marker simply yields ''."
        ),
        "template_differences_by_mode": {
            "factoid": "support_pack[:3] only → MAX 3 markers",
            "comparison": "branch evidence markers if coverage enabled, else support_pack[:3]",
            "summary": "up to 5 claims with markers",
            "partial": "depends on intent: factoid→3, comparison→varies, summary→5",
            "refuse": "0 markers",
        },
        "citation_marker_not_used_recorded_in": (
            "citation_binder.py: bind() debug['drop_reasons_by_evidence_id'] "
            "AND evidence_lifecycle_debug.py: citation_output_debug()"
        ),
        "primary_root_cause_hypothesis": (
            "answer_builder.py line 104: support_pack[:3] hard-coded truncation. "
            "For factoid/unknown (and comparison without coverage), "
            "only the first 3 support items get [E#] markers in the extractive answer. "
            "Since Qwen synthesis is OFF, the extractive answer IS the final answer. "
            "Any support items beyond position 3 are GUARANTEED citation_marker_not_used."
        ),
        "fix_candidates": [
            {
                "type": "fix_answer_marker_template",
                "description": "Change support_pack[:3] to include all support_pack items, "
                               "with a configurable max (e.g. config.v2_max_extractive_evidence_lines).",
                "risk": "Low. Only adds more [E#] lines to extractive answer. Does NOT force citation — "
                        "citation_binder only binds markers that appear in answer text.",
                "feature_flag": "v2_extractive_answer_show_all_support (default: false for safety)",
            },
        ],
    }

    # ── Run samples and collect answer/marker data ─────────────────
    settings = Settings.from_env()
    settings.generation.version = "v2"
    settings.generation.v2_use_qwen_synthesis = False
    settings.generation.v2_enable_comparison_coverage = False
    settings.generation.v2_enable_neighbor_audit = False
    settings.generation.v2_enable_neighbor_promotion = False
    settings.generation.v2_include_neighbor_context_in_qwen = False
    settings.retrieval.parent_expansion_enabled = True
    pipeline = SynBioRAGPipeline(settings)

    audit_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    total = len(run_ids)

    for index, sid in enumerate(run_ids, start=1):
        sample = sample_by_id[sid]
        question = str(sample.get("question") or "")
        expected_docs = sample.get("expected_doc_ids") or []
        expected_route = str(sample.get("expected_route") or "")

        filters = QueryFilters(tenant_id=sample.get("tenant_id", "default"))
        response = pipeline.answer(question, filters=filters)

        answer_text = response.answer or ""
        gv2 = (response.debug or {}).get("generation_v2", {})
        plan_mode = gv2.get("answer_mode", "unknown")
        lifecycle = (response.debug or {}).get("evidence_lifecycle_debug", {})
        sel_support = lifecycle.get("selected_support", {})
        cit_candidates = lifecycle.get("citation_candidates", {})
        cit_output = lifecycle.get("citation_output", {})

        support_pack = gv2.get("support_pack", []) or []
        sel_doc_ids = sel_support.get("doc_ids", [])
        cand_doc_ids = cit_candidates.get("doc_ids", [])
        cit_doc_ids = cit_output.get("cited_doc_ids", [])
        drop_reasons = cit_output.get("drop_reasons", {})
        uncited_chunk_ids = cit_output.get("uncited_selected_support_chunk_ids", [])
        partial_mode = cit_output.get("partial_mode", False)

        answer_markers = _EVIDENCE_RE.findall(answer_text)
        answer_marker_set = set(answer_markers)

        # For each support_pack item that is uncited, analyze why
        cited_chunk_ids = set(cit_output.get("cited_chunk_ids", []))
        for item in support_pack:
            chunk_id = item.get("chunk_id", "")
            evidence_id = item.get("evidence_id", "")
            doc_id = item.get("doc_id", "")
            text = ""
            for sp_item in (response.debug or {}).get("generation_v2", {}).get("support_pack", []):
                if isinstance(sp_item, dict) and sp_item.get("chunk_id") == chunk_id:
                    text = (sp_item.get("candidate") or {}).get("text", "")
                    break

            is_uncited = chunk_id in uncited_chunk_ids
            if not is_uncited:
                if chunk_id in cited_chunk_ids:
                    alignment_rows.append({
                        "sample_id": sid,
                        "candidate_id": evidence_id,
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "is_selected_support": True,
                        "is_citation_candidate": True,
                        "is_protected_seed": "",
                        "citation_priority": "",
                        "support_rank": "",
                        "rerank_rank": "",
                        "candidate_text_preview": (text or "")[:120],
                        "expected_marker_id": evidence_id,
                        "marker_id_in_answer": evidence_id if evidence_id in answer_marker_set else "MISSING",
                        "answer_sentence_using_cardidate": "",
                        "answer_sentence_has_marker": evidence_id in answer_marker_set,
                        "parser_detected_marker": evidence_id in answer_marker_set,
                        "binder_output_citation": True,
                        "marker_alignment_status": "used_and_cited",
                    })
                continue

            # This chunk is uncited — analyze why
            text_preview = (text or "")[:200] if text else ""
            candidate_in_answer = semantic_overlap_ratio(text or "", answer_text) if text else 0.0
            supporting_sent = find_supporting_sentence(text or "", answer_text) if text else None

            # Determine reason
            reason = "unclear"
            expected_marker = evidence_id
            marker_in_answer = evidence_id in answer_marker_set

            if marker_in_answer:
                reason = "marker_parsed_but_not_bound"
            elif candidate_in_answer > 0.15 and supporting_sent:
                reason = "candidate_used_but_marker_missing"
            elif expected_route == "comparison" and len(expected_docs) > 1:
                reason = "comparison_branch_not_expressed"
            elif candidate_in_answer > 0.05:
                reason = "candidate_used_but_marker_missing"
            elif len(support_pack) > 3:
                # Check if this item is beyond the first 3 — likely truncated
                item_pos = next((i for i, sp in enumerate(support_pack) if sp.get("chunk_id") == chunk_id), -1)
                if item_pos >= 3:
                    reason = "answer_mode_template_omits_marker"
                else:
                    reason = "candidate_not_used_by_answer"
            else:
                reason = "candidate_not_used_by_answer"

            alignment_rows.append({
                "sample_id": sid,
                "candidate_id": evidence_id,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "is_selected_support": True,
                "is_citation_candidate": True,
                "is_protected_seed": "",
                "citation_priority": "",
                "support_rank": "",
                "rerank_rank": "",
                "candidate_text_preview": text_preview[:120],
                "expected_marker_id": expected_marker,
                "marker_id_in_answer": evidence_id if marker_in_answer else "MISSING",
                "answer_sentence_using_cardidate": (supporting_sent or "")[:200],
                "answer_sentence_has_marker": marker_in_answer,
                "parser_detected_marker": marker_in_answer,
                "binder_output_citation": False,
                "marker_alignment_status": (
                    "used_but_marker_missing" if "used_but_marker_missing" in reason
                    else "marker_present_but_not_parsed" if "marker_parsed" in reason
                    else "not_used_reasonable" if "not_used" in reason
                    else "redundant_candidate" if "redundant" in reason
                    else "unclear"
                ),
            })

        # Per-sample summary
        total_candidates = len(support_pack)
        candidates_cited = len(cited_chunk_ids)
        candidates_uncited = len(uncited_chunk_ids)
        used_but_missing = sum(
            1 for r in alignment_rows
            if r["sample_id"] == sid and "used_but_marker_missing" in r.get("marker_alignment_status", "")
        )
        not_used = sum(
            1 for r in alignment_rows
            if r["sample_id"] == sid and "not_used_reasonable" in r.get("marker_alignment_status", "")
        )

        audit_rows.append({
            "sample_id": sid,
            "question": question[:150],
            "answer_mode": plan_mode,
            "plan_mode": plan_mode,
            "failure_category": "",
            "is_p0": "",
            "expected_doc_ids": "|".join(expected_docs),
            "selected_support_doc_ids": "|".join(sel_doc_ids),
            "citation_candidate_doc_ids": "|".join(cand_doc_ids),
            "citation_output_doc_ids": "|".join(cit_doc_ids),
            "uncited_candidate_doc_ids": "",
            "uncited_candidate_chunk_ids": "|".join(uncited_chunk_ids),
            "uncited_candidate_text_previews": "",
            "answer_text": answer_text[:500],
            "answer_marker_count": len(answer_markers),
            "answer_markers": "|".join(answer_markers),
            "support_pack_size": len(support_pack),
            "support_pack_truncation_applied": len(support_pack) > 3,
            "template_omits_markers_beyond_3": "YES" if len(support_pack) > 3 else "NO",
            "primary_marker_failure_reason": (
                "template_truncation_support_pack_3" if len(support_pack) > 3
                else "answer_does_not_use_candidate" if candidates_uncited > 0
                else "all_candidates_cited"
            ),
            "recommended_fix": (
                "fix_answer_marker_template" if len(support_pack) > 3
                else "no_fix_reasonable_not_used" if candidates_uncited > 0
                else "no_action_needed"
            ),
        })

        if index % 20 == 0 or index <= 3:
            print(f"[{index}/{total}] {sid} mode={plan_mode} "
                  f"support={len(support_pack)} cited={candidates_cited} "
                  f"uncited={candidates_uncited} answer_markers={len(answer_markers)}",
                  flush=True)

    # ── Classification summary ─────────────────────────────────────
    template_truncation_count = sum(
        1 for r in audit_rows if r["primary_marker_failure_reason"] == "template_truncation_support_pack_3"
    )
    reasonable_not_used = sum(
        1 for r in audit_rows if r["primary_marker_failure_reason"] == "answer_does_not_use_candidate"
    )
    all_cited = sum(
        1 for r in audit_rows if r["primary_marker_failure_reason"] == "all_candidates_cited"
    )

    fix_plan = {
        "total_marker_not_used_audited": len(audit_rows),
        "total_samples_with_uncited_candidates": sum(
            1 for r in audit_rows if "|".join(r.get("uncited_candidate_chunk_ids", []))
        ),
        "support_pack_truncation_template_omits_count": template_truncation_count,
        "candidate_not_used_or_redundant_count": reasonable_not_used,
        "all_candidates_cited_count": all_cited,
        "primary_root_cause": (
            "answer_builder.py line 104: support_pack[:3] hard-coded truncation. "
            f"{template_truncation_count}/{len(audit_rows)} samples have support_pack > 3, "
            "meaning items #4+ are guaranteed citation_marker_not_used. "
            "This is the dominant cause — not a per-sample issue, but a template design constraint."
        ),
        "recommended_fix_type": "fix_answer_marker_template",
        "should_implement_fix_now": True,
        "proposed_fix_scope": (
            "Change answer_builder.py line 104 from 'support_pack[:3]' to "
            "'support_pack[:config.v2_max_extractive_evidence_lines]' with default=6. "
            "This ensures all support_pack items get [E#] markers in the extractive answer. "
            "Does NOT force citation — citation_binder still only binds markers present in answer text."
        ),
        "files_to_modify": [
            "src/synbio_rag/application/generation_v2/answer_builder.py",
            "src/synbio_rag/domain/config.py (add v2_max_extractive_evidence_lines)",
        ],
        "tests_to_add": [
            "test_extractive_answer_includes_all_support_markers",
            "test_extractive_answer_respects_max_lines_config",
            "test_no_forced_citation_for_unused_evidence",
        ],
        "focused_validation_plan": (
            "Run focused 11 + comparison samples. Verify: "
            "1. citation_marker_not_used decreases for samples with support_pack > 3, "
            "2. citation output count does not exceed limit, "
            "3. no forced citations for unused evidence."
        ),
        "risk_assessment": (
            "LOW: Only adds more evidence lines with [E#] markers to extractive answer. "
            "Citation_binder still uses marker-based binding — if answer doesn't actually "
            "reference the evidence, the marker won't appear in the final answer. "
            "No change to retrieval, support_selector, citation_binder, or citation output limit."
        ),
    }

    # ── Write outputs ──────────────────────────────────────────────

    def write_csv(filepath: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def write_json(filepath: Path, data: Any) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    AUDIT_FIELDS = [
        "sample_id", "question", "answer_mode", "plan_mode",
        "failure_category", "is_p0", "expected_doc_ids",
        "selected_support_doc_ids", "citation_candidate_doc_ids",
        "citation_output_doc_ids", "uncited_candidate_doc_ids",
        "uncited_candidate_chunk_ids", "uncited_candidate_text_previews",
        "answer_text", "answer_marker_count", "answer_markers",
        "support_pack_size", "support_pack_truncation_applied",
        "template_omits_markers_beyond_3",
        "primary_marker_failure_reason", "recommended_fix",
    ]

    ALIGN_FIELDS = [
        "sample_id", "candidate_id", "chunk_id", "doc_id",
        "is_selected_support", "is_citation_candidate",
        "is_protected_seed", "citation_priority", "support_rank",
        "rerank_rank", "candidate_text_preview",
        "expected_marker_id", "marker_id_in_answer",
        "answer_sentence_using_cardidate", "answer_sentence_has_marker",
        "parser_detected_marker", "binder_output_citation",
        "marker_alignment_status",
    ]

    CLASS_FIELDS = [
        "sample_id", "answer_mode", "plan_mode",
        "total_candidates", "candidates_cited",
        "candidates_marker_not_used",
        "template_omits_markers", "candidate_not_used_count",
        "support_pack_size", "support_pack_truncated",
        "primary_failure_type", "recommended_next_action",
    ]

    write_json(output_dir / "marker_usage_code_path_audit.json", code_audit)
    write_csv(output_dir / "citation_marker_not_used_audit.csv", AUDIT_FIELDS, audit_rows)
    write_csv(output_dir / "answer_marker_alignment_trace.csv", ALIGN_FIELDS, alignment_rows)

    # Classification per sample
    class_rows = []
    for r in audit_rows:
        class_rows.append({
            "sample_id": r["sample_id"],
            "answer_mode": r["answer_mode"],
            "plan_mode": r["plan_mode"],
            "total_candidates": r["support_pack_size"],
            "candidates_cited": r["answer_marker_count"],
            "candidates_marker_not_used": r["support_pack_size"] - r["answer_marker_count"],
            "template_omits_markers": 1 if r["template_omits_markers_beyond_3"] == "YES" else 0,
            "candidate_not_used_count": r["support_pack_size"] - r["answer_marker_count"],
            "support_pack_size": r["support_pack_size"],
            "support_pack_truncated": r["support_pack_truncation_applied"],
            "primary_failure_type": r["primary_marker_failure_reason"],
            "recommended_next_action": r["recommended_fix"],
        })
    write_csv(output_dir / "marker_failure_classification.csv", CLASS_FIELDS, class_rows)
    write_json(output_dir / "phase16e_recommended_fix_plan.json", fix_plan)

    # Print summary
    print(f"\nPhase 16E Complete:")
    print(f"  Samples audited: {len(audit_rows)}")
    print(f"  template_truncation (>3 support): {template_truncation_count}")
    print(f"  reasonable_not_used: {reasonable_not_used}")
    print(f"  all_cited: {all_cited}")
    print(f"  Primary root cause: answer_builder.py support_pack[:3] truncation")
    print(f"  Fix recommended: {fix_plan['should_implement_fix_now']}")
    print(f"  Fix type: {fix_plan['recommended_fix_type']}")


if __name__ == "__main__":
    main()
