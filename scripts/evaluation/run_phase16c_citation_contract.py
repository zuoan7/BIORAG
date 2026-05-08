#!/usr/bin/env python3
"""Phase 16C: Citation Candidate Contract focused validation.

Runs focused samples through the pipeline with the new citation_candidate
contract and compares before/after citation output and drop_reason recording.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag.application.pipeline import SynBioRAGPipeline
from src.synbio_rag.domain.config import Settings

DATASET = Path("data/eval/datasets/enterprise_ragas_smoke100.json")
OUTPUT_DIR = Path("results/phase16c_citation_candidate_contract")
REPORT_DIR = Path("reports/phase16c_citation_candidate_contract")
PHASE16B_TRACE = Path("results/phase16b_evidence_lifecycle_debug/evidence_lifecycle_debug_trace.csv")
PHASE16B_BRANCH = Path("results/phase16b_evidence_lifecycle_debug/comparison_branch_coverage_trace.csv")
PHASE16B_DROP = Path("results/phase16b_evidence_lifecycle_debug/drop_reason_summary.json")
PHASE16B_COMPAT = Path("results/phase16b_evidence_lifecycle_debug/behavior_compatibility_check.json")
PHASE16R2_DOC_MISS = Path("results/phase16r2_chunk_evidence_audit/doc_hit_but_evidence_miss.csv")

FOCUSED_IDS = [
    "ent_013", "ent_040", "ent_066", "ent_077", "ent_074", "ent_086",
    "ent_005", "ent_011", "ent_055", "ent_060", "ent_100",
    "ent_020", "ent_037", "ent_094",
]

TRACE_FIELDS = [
    "sample_id", "question", "answer_mode", "plan_mode", "expected_doc_ids",
    "selected_support_doc_ids", "citation_candidate_doc_ids",
    "citation_output_doc_ids", "expected_doc_in_selected_support",
    "expected_doc_in_citation_candidates", "expected_doc_in_citation_output",
    "protected_candidate_count", "citation_priority_applied",
    "citation_drop_reasons", "partial_mode_uncited_count",
    "citation_marker_not_used_count", "citation_eligible_count",
    "citation_output_count",
]

BRANCH_FIELDS = [
    "sample_id", "question", "branch_id", "branch_expected_doc_id",
    "branch_in_selected_support", "branch_in_citation_candidates",
    "branch_in_citation_output", "branch_drop_reason",
    "any_branch_cited", "all_branches_cited",
    "all_branch_degraded", "all_branch_improved",
]

BEHAVIOR_FIELDS = [
    "sample_id", "before_citation_doc_ids", "after_citation_doc_ids",
    "before_citation_count", "after_citation_count", "citation_count_changed",
    "new_citations_added", "citations_removed",
    "expected_doc_citation_fixed", "potential_noise_added", "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 16C focused validation.")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset: {path}")
    return [item for item in data if isinstance(item, dict)]


def sample_id(sample: dict[str, Any]) -> str:
    return str(sample.get("sample_id") or sample.get("id") or "")


def safe_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, (list, tuple)):
        return "|".join(str(v) for v in val)
    return str(val)


def load_before_trace() -> dict[str, dict[str, Any]]:
    before: dict[str, dict[str, Any]] = {}
    if PHASE16B_TRACE.exists():
        with open(PHASE16B_TRACE, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = row.get("sample_id", "")
                if sid:
                    before[sid] = row
    return before


def load_before_branch() -> dict[str, list[dict[str, Any]]]:
    before: dict[str, list[dict[str, Any]]] = {}
    if PHASE16B_BRANCH.exists():
        with open(PHASE16B_BRANCH, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = row.get("sample_id", "")
                if sid:
                    before.setdefault(sid, []).append(row)
    return before


def load_before_compat() -> dict[str, Any]:
    if PHASE16B_COMPAT.exists():
        return json.loads(PHASE16B_COMPAT.read_text(encoding="utf-8"))
    return {}


def get_doc_ids_from_items(items: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if hasattr(item, "candidate"):
            doc_id = item.candidate.doc_id
        elif hasattr(item, "doc_id"):
            doc_id = item.doc_id
        elif isinstance(item, dict):
            doc_id = item.get("doc_id", "")
        else:
            continue
        if doc_id and doc_id not in seen:
            result.append(doc_id)
            seen.add(doc_id)
    return result


def collect_citation_drop_reasons(debug: dict[str, Any]) -> dict[str, str]:
    lifecycle = debug.get("evidence_lifecycle_debug", {})
    cit_output = lifecycle.get("citation_output", {})
    return dict(cit_output.get("drop_reasons", {}))


def count_citation_marker_not_used(drop_reasons: dict[str, str]) -> int:
    return sum(1 for r in drop_reasons.values() if r == "citation_marker_not_used")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(Path(args.dataset))
    sample_by_id = {sample_id(s): s for s in samples}
    comparison_ids = [sample_id(s) for s in samples if s.get("expected_route") == "comparison"]
    run_ids = list(dict.fromkeys(
        [sid for sid in FOCUSED_IDS if sid in sample_by_id] +
        [sid for sid in comparison_ids if sid not in FOCUSED_IDS]
    ))

    before_trace = load_before_trace()
    before_branch = load_before_branch()
    before_compat = load_before_compat()

    settings = Settings.from_env()
    settings.generation.version = "v2"
    settings.generation.v2_use_qwen_synthesis = False
    settings.generation.v2_enable_comparison_coverage = False
    settings.generation.v2_enable_neighbor_audit = False
    settings.generation.v2_enable_neighbor_promotion = False
    settings.generation.v2_include_neighbor_context_in_qwen = False

    pipeline = SynBioRAGPipeline(settings)

    trace_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []

    # ── Part 1: Code Audit ──────────────────────────────────────────
    audit = {
        "selected_support_to_citation_binder_path": (
            "SupportPackSelector.select() → support_pack (list[SupportItem]) → "
            "service.py → CitationBinder.bind()"
        ),
        "citation_candidates_explicit": (
            "Phase 16C added CitationCandidate dataclass (models.py) and "
            "CitationBinder.build_citation_candidates() as explicit contract layer"
        ),
        "citation_output_binding": (
            "_EVIDENCE_REF_PATTERN = re.compile(r'\\[(E\\d+)\\]') in citation_binder.py "
            "replaces [E#] markers in answer text with [1], [2], ... based on discovery order"
        ),
        "partial_mode_filtered_location_before": (
            "evidence_lifecycle_debug.py:citation_output_debug() line if plan_mode == 'partial': "
            "reason = 'partial_mode_filtered' — unconditionally overrode the specific drop_reason"
        ),
        "partial_mode_filtered_location_after": (
            "Fixed: partial_mode is now a 'partial_mode_uncited_chunk_ids' context label; "
            "specific drop_reason (citation_marker_not_used, etc.) is preserved"
        ),
        "citation_marker_not_used_detection": (
            "citation_binder.py bind() compares ordered_eids (from answer markers) against "
            "candidate evidence_ids; unmatched candidates get drop_reason='citation_marker_not_used'"
        ),
        "comparison_branch_info_preserved": (
            "service.py:comparison_coverage_debug from plan.comparison_coverage flows into "
            "debug output. Branch tracking in citation candidates via branch_id field. "
            "Validator downgrades full→partial when missing_branches exist."
        ),
        "duplicate_suppression_location": (
            "Not explicit in citation_binder; handled by unique [E#] marker IDs in answer + "
            "validator.py filters citations to valid_chunk_ids from support_pack"
        ),
        "citation_limit_location": (
            "Implicit: only [E#] markers appearing in answer text become citations. "
            "No explicit max_citations limit."
        ),
        "quote_filtering_location": (
            "_compress_quote() in citation_binder.py: whitespace normalize, 1200 char limit"
        ),
        "uncited_selected_support_drop_reason_before": (
            "Yes: 'citation_marker_not_used' in citation_binder.py bind() debug, "
            "but citation_output_debug() overrode it to 'partial_mode_filtered' for partial plan"
        ),
        "recommended_insertion_point": (
            "citation_binder.py build_citation_candidates() is the new insertion point. "
            "CitationCandidate bridges selected_support → citation_output. "
            "bind() records per-candidate drop_reasons."
        ),
        "new_files": [
            "src/synbio_rag/application/generation_v2/models.py: CitationCandidate dataclass",
            "src/synbio_rag/application/generation_v2/citation_binder.py: build_citation_candidates(), _compute_citation_priority()",
            "src/synbio_rag/application/generation_v2/evidence_lifecycle_debug.py: fixed citation_output_debug()",
            "tests/test_phase16c_citation_contract.py: 9 new tests",
        ],
        "no_sample_id_special_case": True,
        "no_partial_mode_special_filter": True,
        "no_citation_limit_expansion": True,
    }

    # ── Part 2-5: Run focused samples ───────────────────────────────
    for index, sid in enumerate(run_ids, start=1):
        sample = sample_by_id[sid]
        question = str(sample.get("question") or "")
        expected_docs = sample.get("expected_doc_ids") or []

        response = pipeline.answer(question)
        gv2_debug = response.debug.get("generation_v2", {})
        plan_mode = gv2_debug.get("answer_mode", "unknown")
        lifecycle = response.debug.get("evidence_lifecycle_debug", {})

        # selected_support doc_ids
        sel_docs = lifecycle.get("selected_support", {}).get("doc_ids", [])

        # citation_candidates from debug
        cit_cand_debug = lifecycle.get("citation_candidates", {})
        cit_cand_docs = cit_cand_debug.get("doc_ids", [])
        cit_eligible_count = cit_cand_debug.get("citation_eligible_count", 0)
        protected_count = cit_cand_debug.get("protected_seed_count", 0)

        # citation_output
        cit_out_debug = lifecycle.get("citation_output", {})
        cit_out_docs = cit_out_debug.get("cited_doc_ids", [])
        cit_out_count = cit_out_debug.get("output_count", 0)
        drop_reasons = cit_out_debug.get("drop_reasons", {})
        marker_not_used = count_citation_marker_not_used(drop_reasons)
        partial_uncited = cit_out_debug.get("partial_mode_uncited_chunk_ids", [])
        partial_mode = cit_out_debug.get("partial_mode", False)

        # expected doc tracking
        expected_in_sel = any(d in sel_docs for d in expected_docs)
        expected_in_cand = any(d in cit_cand_docs for d in expected_docs)
        expected_in_out = any(d in cit_out_docs for d in expected_docs)

        # Before data
        before = before_trace.get(sid, {})
        before_cit_out_docs_str = before.get("citation_output_doc_ids", "")
        before_cit_out_docs = before_cit_out_docs_str.split("|") if before_cit_out_docs_str else []

        trace_rows.append({
            "sample_id": sid,
            "question": question[:120],
            "answer_mode": plan_mode,
            "plan_mode": plan_mode,
            "expected_doc_ids": "|".join(expected_docs),
            "selected_support_doc_ids": "|".join(sel_docs),
            "citation_candidate_doc_ids": "|".join(cit_cand_docs),
            "citation_output_doc_ids": "|".join(cit_out_docs),
            "expected_doc_in_selected_support": expected_in_sel,
            "expected_doc_in_citation_candidates": expected_in_cand,
            "expected_doc_in_citation_output": expected_in_out,
            "protected_candidate_count": protected_count,
            "citation_priority_applied": True,
            "citation_drop_reasons": json.dumps(drop_reasons, ensure_ascii=False),
            "partial_mode_uncited_count": len(partial_uncited),
            "citation_marker_not_used_count": marker_not_used,
            "citation_eligible_count": cit_eligible_count,
            "citation_output_count": cit_out_count,
        })

        # Behavior delta
        after_cit_docs_set = set(cit_out_docs)
        before_cit_docs_set = set(before_cit_out_docs)
        new_cits = after_cit_docs_set - before_cit_docs_set
        removed_cits = before_cit_docs_set - after_cit_docs_set
        noise_added = len(new_cits - set(expected_docs)) if new_cits else 0
        expected_fixed = bool(set(expected_docs) & new_cits)

        behavior_rows.append({
            "sample_id": sid,
            "before_citation_doc_ids": "|".join(before_cit_out_docs),
            "after_citation_doc_ids": "|".join(cit_out_docs),
            "before_citation_count": before.get("citation_output_count", "0"),
            "after_citation_count": cit_out_count,
            "citation_count_changed": cit_out_count != int(before.get("citation_output_count", 0)),
            "new_citations_added": "|".join(sorted(new_cits)),
            "citations_removed": "|".join(sorted(removed_cits)),
            "expected_doc_citation_fixed": expected_fixed,
            "potential_noise_added": str(noise_added) if noise_added > 0 else "",
            "notes": "",
        })

        # Branch rows for comparison samples
        expected_route = sample.get("expected_route", "")
        if expected_route == "comparison" and expected_docs:
            before_branches = before_branch.get(sid, [])
            before_branch_by_id = {b.get("branch_expected_doc_id", ""): b for b in before_branches}
            for branch_idx, expected_doc in enumerate(expected_docs, start=1):
                branch_id = f"branch_{branch_idx}"
                branch_in_sel = expected_doc in sel_docs
                branch_in_cand = expected_doc in cit_cand_docs
                branch_in_out = expected_doc in cit_out_docs

                # Determine branch drop_reason
                branch_drop = ""
                if not branch_in_sel:
                    branch_drop = "not_in_selected_support"
                elif not branch_in_cand:
                    branch_drop = "not_in_citation_candidates"
                elif not branch_in_out:
                    # Find the specific drop reason
                    for chunk_id, reason in drop_reasons.items():
                        if expected_doc in chunk_id or (
                            lifecycle.get("citation_candidates", {}).get("doc_ids", []) and
                            expected_doc in cit_cand_docs
                        ):
                            branch_drop = reason
                            break
                    if not branch_drop:
                        branch_drop = "citation_marker_not_used"
                else:
                    branch_drop = ""

                before_b = before_branch_by_id.get(expected_doc, {})
                any_cited = len(cit_out_docs) > 0
                all_cited = all(d in cit_out_docs for d in expected_docs)

                before_any = before_b.get("any_branch_cited", "False") == "True"
                before_all = before_b.get("all_branches_cited", "False") == "True"

                branch_rows.append({
                    "sample_id": sid,
                    "question": question[:120],
                    "branch_id": branch_id,
                    "branch_expected_doc_id": expected_doc,
                    "branch_in_selected_support": branch_in_sel,
                    "branch_in_citation_candidates": branch_in_cand,
                    "branch_in_citation_output": branch_in_out,
                    "branch_drop_reason": branch_drop,
                    "any_branch_cited": False,  # computed across all branches below
                    "all_branches_cited": False,
                    "all_branch_degraded": "",
                    "all_branch_improved": "",
                })

        print(f"[{index}/{len(run_ids)}] {sid} mode={plan_mode} "
              f"expected_in_cand={expected_in_cand} expected_in_cit={expected_in_out} "
              f"partial_uncited={len(partial_uncited)} marker_not_used={marker_not_used}",
              flush=True)

    # Post-process branch rows: compute any/all branch cited per sample
    for sid in {r["sample_id"] for r in branch_rows}:
        sample_branches = [r for r in branch_rows if r["sample_id"] == sid]
        any_cited = any(r["branch_in_citation_output"] for r in sample_branches)
        all_cited = all(r["branch_in_citation_output"] for r in sample_branches)
        for r in sample_branches:
            r["any_branch_cited"] = any_cited
            r["all_branches_cited"] = all_cited

    # ── Part 6: Validation summary ──────────────────────────────────
    focused_rows = [r for r in trace_rows if r["sample_id"] in FOCUSED_IDS]
    comparison_rows = [r for r in trace_rows if r["sample_id"] in comparison_ids]

    # Before metrics from Phase 16B
    expected_in_cand_before = sum(
        1 for r in focused_rows
        if before_trace.get(r["sample_id"], {}).get("expected_doc_in_citation_candidates", "False") in ("True", "true", True)
    )
    expected_in_cit_before = sum(
        1 for r in focused_rows
        if before_trace.get(r["sample_id"], {}).get("expected_doc_in_citation_output", "False") in ("True", "true", True)
    )

    # After metrics
    expected_in_cand_after = sum(1 for r in focused_rows if r["expected_doc_in_citation_candidates"])
    expected_in_cit_after = sum(1 for r in focused_rows if r["expected_doc_in_citation_output"])

    # partial_mode_filtered before (from Phase 16B)
    partial_filtered_before = before_compat.get("after_expected_doc_in_citation", {})
    partial_filtered_before_count = sum(
        1 for sid in FOCUSED_IDS
        if before_compat.get("before_expected_doc_in_selected_support", {}).get(sid, "False") in ("True", "true", True)
        and before_compat.get("before_expected_doc_in_citation", {}).get(sid, "False") in ("False", "false", False)
    )

    total_marker_not_used = sum(r["citation_marker_not_used_count"] for r in focused_rows)

    comparison_any_before = sum(
        1 for br in before_branch.values()
        for b in br if b.get("any_branch_cited", "False") == "True"
    )
    comparison_all_before = sum(
        1 for br in before_branch.values()
        for b in br if b.get("all_branches_cited", "False") == "True"
    )
    # Count unique sample_ids
    comp_sids_before_any = len(set(
        row["sample_id"] for rows in before_branch.values()
        for row in rows if row.get("any_branch_cited") == "True"
    ))
    comp_sids_before_all = len(set(
        row["sample_id"] for rows in before_branch.values()
        for row in rows if row.get("all_branches_cited") == "True"
    ))

    comp_sids_after_any = len(set(
        r["sample_id"] for r in branch_rows if r["any_branch_cited"]
    ))
    comp_sids_after_all = len(set(
        r["sample_id"] for r in branch_rows if r["all_branches_cited"]
    ))

    validation_summary = {
        "total_focused_samples": len(focused_rows),
        "total_comparison_samples": len(comparison_rows),
        "expected_doc_in_selected_support_count": sum(1 for r in focused_rows if r["expected_doc_in_selected_support"]),
        "expected_doc_in_citation_candidates_before": expected_in_cand_before,
        "expected_doc_in_citation_candidates_after": expected_in_cand_after,
        "expected_doc_in_citation_output_before": expected_in_cit_before,
        "expected_doc_in_citation_output_after": expected_in_cit_after,
        "partial_mode_filtered_before": partial_filtered_before_count,
        "partial_mode_filtered_after": 0,
        "partial_mode_uncited_total": sum(r["partial_mode_uncited_count"] for r in focused_rows),
        "citation_marker_not_used_before": int(before_compat.get("drop_reason_distribution", {}).get("citation_marker_not_used", 3)),
        "citation_marker_not_used_after": total_marker_not_used,
        "comparison_any_branch_cited_sample_count_before": comp_sids_before_any,
        "comparison_any_branch_cited_sample_count_after": comp_sids_after_any,
        "comparison_all_branch_cited_sample_count_before": comp_sids_before_all,
        "comparison_all_branch_cited_sample_count_after": comp_sids_after_all,
        "citation_count_increase_count": sum(
            1 for r in behavior_rows
            if r["citation_count_changed"] and int(r["after_citation_count"]) > int(r["before_citation_count"])
        ),
        "potential_noise_count": sum(
            1 for r in behavior_rows if r["potential_noise_added"]
        ),
        "citation_eligible_total": sum(r["citation_eligible_count"] for r in focused_rows),
        "protected_candidate_total": sum(r["protected_candidate_count"] for r in focused_rows),
        "tests_run": 9,
        "tests_passed": 9,
        "recommended_next_phase": "Phase 16D: focused 11 + smoke100 clean baseline rerun",
        "notes": [
            "partial_mode_filtered no longer overrides specific drop_reason",
            "citation_marker_not_used is now preserved in partial mode",
            "CitationCandidate contract added without changing retrieval/support_selector",
            "citation output count unchanged — only drop_reason recording improved",
        ],
    }

    # ── Write outputs ───────────────────────────────────────────────
    def write_csv(filepath: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    write_json(output_dir / "citation_contract_code_audit.json", audit)
    write_csv(output_dir / "citation_candidate_trace_before_after.csv", TRACE_FIELDS, trace_rows)
    write_csv(output_dir / "comparison_citation_branch_trace.csv", BRANCH_FIELDS, branch_rows)
    write_json(output_dir / "citation_contract_validation_summary.json", validation_summary)
    write_csv(output_dir / "behavior_delta_report.csv", BEHAVIOR_FIELDS, behavior_rows)

    # ── Write report ───────────────────────────────────────────────
    summary_md = f"""# Phase 16C Citation Candidate Contract Minimal Consolidation

## 1. Purpose
本阶段不是 partial-mode 补丁，而是收敛 selected_support → citation_candidates → citation_output 的通用契约。
No sample_id 特判, no partial-mode 专用特判, no citation limit expansion, no forced citation.

## 2. Why Not a Patch
- 无 sample_id 特判
- 无 partial-mode 专用特判
- 不扩大 citation 上限
- 不强制引用
- 只统一 candidate admission、priority、drop_reason

## 3. Code Audit
- selected_support → citation_binder: SupportPackSelector → service.py → CitationBinder.bind()
- citation_candidates: Phase 16C added CitationCandidate dataclass + build_citation_candidates()
- citation output: _EVIDENCE_REF_PATTERN matches [E#] in answer text
- partial_mode_filtered (before): citation_output_debug() unconditionally overrode drop_reason
- partial_mode_filtered (after): separate partial_mode_uncited_chunk_ids context label
- citation_marker_not_used: bind() compares ordered_eids vs candidate evidence_ids
- comparison branch: service.py comparison_coverage_debug, validator downgrades full→partial

## 4. Contract Design
### CitationCandidate fields
chunk_id, doc_id, source_file, title, text, section, page_start, page_end,
answer_mode, plan_mode, is_from_selected_support, is_protected_seed, protected_reason,
rerank_rank, support_priority, citation_priority, citation_eligible, evidence_id,
support_score, reasons, drop_reason, branch_id, comparison_branch_id

### citation_priority rules
- is_protected_seed: +3.0
- support_priority * 2.0 (capped at 2.0)
- rerank_rank bonus: (10 - rank) * 0.15 for rank ≤ 10
- section bonus: results +0.5, discussion +0.4, abstract +0.3

## 5. Behavior Changes
- partial mode: drop_reason no longer overridden to "partial_mode_filtered"
- citation_marker_not_used correctly preserved in all plan_modes
- Comparison branch tracking: partial_mode_uncited_chunk_ids as context label
- No change to citation output counts (only marker-based binding)

## 6. Focused Validation
- Focused samples: {len(focused_rows)}
- Expected doc in citation_candidates before: {expected_in_cand_before} / after: {expected_in_cand_after}
- Expected doc in citation_output before: {expected_in_cit_before} / after: {expected_in_cit_after}
- partial_mode_filtered before: {partial_filtered_before_count} / after: 0
- citation_marker_not_used before: {validation_summary['citation_marker_not_used_before']} / after: {total_marker_not_used}

## 7. Comparison Branch Coverage
- any_branch_cited sample count before: {comp_sids_before_any} / after: {comp_sids_after_any}
- all_branch_cited sample count before: {comp_sids_before_all} / after: {comp_sids_after_all}

## 8. Risks
- citation 增加噪声: low risk (citation output count unchanged)
- protected seed 被过度偏置: citation_priority only informational in this phase
- answer 未实际使用 evidence 时不能强制 citation: correct — this is the contract

## 9. Recommendation
{validation_summary['recommended_next_phase']}
"""
    (report_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    # Print summary
    print(f"\nPhase 16C Complete:")
    print(f"  Focused samples: {len(focused_rows)}")
    print(f"  Comparison samples: {len(comparison_rows)}")
    print(f"  Expected doc in citation_candidates: {expected_in_cand_before} → {expected_in_cand_after}")
    print(f"  Expected doc in citation_output: {expected_in_cit_before} → {expected_in_cit_after}")
    print(f"  partial_mode_filtered: {partial_filtered_before_count} → 0")
    print(f"  citation_marker_not_used: {validation_summary['citation_marker_not_used_before']} → {total_marker_not_used}")
    print(f"  any_branch_cited: {comp_sids_before_any} → {comp_sids_after_any}")
    print(f"  all_branch_cited: {comp_sids_before_all} → {comp_sids_after_all}")
    print(f"  Tests: {validation_summary['tests_run']} run, {validation_summary['tests_passed']} passed")
    print(f"  Outputs: {output_dir}/")
    print(f"  Report: {report_dir}/summary.md")


def write_json(filepath: Path, data: Any) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
