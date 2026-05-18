#!/usr/bin/env python3
"""Build Phase7R production table index proposal artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports/v7_phase7_table_index_production_proposal"
DATA_DIR = ROOT / "data/experiments/v7_phase7_table_index_production_proposal"
RESULTS_DIR = ROOT / "results/v7_phase7_table_index_production_proposal"

REPORT_PATHS = [
    "phase7r_guardrail.md",
    "phase7r_blocker_review.md",
    "production_table_index_build_proposal.md",
    "production_index_artifact_manifest.md",
    "promotion_gate_matrix.md",
    "promotion_rollback_design.md",
    "citation_readiness_coupling.md",
    "canary_shadow_rollout_plan.md",
    "risk_register.md",
    "phase7r_validation_report.md",
    "phase7r_summary.md",
]

STRUCTURED_PATHS = [
    "production_index_artifact_manifest_template.json",
    "promotion_gate_matrix.csv",
    "rollback_scenario_matrix.csv",
    "rollout_stage_matrix.csv",
    "risk_register.csv",
]

RESULT_PATHS = ["phase7r_validation_results.csv"]


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = [str(row.get(field, "")).replace("|", "\\|") for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def promotion_gate_rows() -> list[dict[str, str]]:
    return [
        {
            "gate_id": "canonical_paper_source_resolved",
            "gate_name": "Canonical paper source resolved",
            "required_for_promotion": "true",
            "current_preview_status": "fail",
            "blocks_current_preview_units": "true",
            "block_reason": "canonical_source_unresolved",
            "required_evidence": "canonical_source_manifest_id resolves doc_id to paper title, canonical paper source, DOI/PMID when available, and page scope",
            "owner_phase": "Phase7S",
            "notes": "Current table artifacts must not replace the canonical paper source with CSV/crop/debug paths.",
        },
        {
            "gate_id": "csv_crop_path_not_formal_source",
            "gate_name": "CSV/crop path not formal source",
            "required_for_promotion": "true",
            "current_preview_status": "conditional_pass",
            "blocks_current_preview_units": "false",
            "block_reason": "debug_path_formal_source_misuse",
            "required_evidence": "canonical_source.source_file differs from source_csv_path, source_pdf_crop_path, source_markdown_path, and debug artifact extensions",
            "owner_phase": "Phase7S",
            "notes": "CSV/crop/markdown paths remain provenance_debug only.",
        },
        {
            "gate_id": "table_id_caption_page_valid",
            "gate_name": "Table id, caption, and page valid",
            "required_for_promotion": "true",
            "current_preview_status": "needs_review",
            "blocks_current_preview_units": "true",
            "block_reason": "table_scope_not_production_verified",
            "required_evidence": "table_id, table_caption, page_start, and page_end are present and verified against canonical paper source",
            "owner_phase": "Phase7S",
            "notes": "Preview scope can be used for debug, not production promotion.",
        },
        {
            "gate_id": "table_row_cell_group_scope_valid",
            "gate_name": "Table/row/cell-group scope valid",
            "required_for_promotion": "true",
            "current_preview_status": "needs_review",
            "blocks_current_preview_units": "true",
            "block_reason": "scope_binding_not_production_verified",
            "required_evidence": "table_unit_type maps to citation_scope in table, row, or cell_group with row/header requirements satisfied",
            "owner_phase": "Phase7S",
            "notes": "Scope validity must be independently reviewed, not inferred from retrieval text.",
        },
        {
            "gate_id": "citation_scope_not_value",
            "gate_name": "Citation scope is not value",
            "required_for_promotion": "true",
            "current_preview_status": "pass",
            "blocks_current_preview_units": "false",
            "block_reason": "citation_scope_value_forbidden",
            "required_evidence": "typed citation schema excludes value from citation_scope",
            "owner_phase": "Phase7S",
            "notes": "A value scope blocks promotion.",
        },
        {
            "gate_id": "value_level_citation_disabled_unless_value_bboxes_verified",
            "gate_name": "Value-level citation disabled unless value bboxes verified",
            "required_for_promotion": "true",
            "current_preview_status": "conditional_pass",
            "blocks_current_preview_units": "false",
            "block_reason": "value_bbox_unverified_value_claim",
            "required_evidence": "value_level_citation_claim_allowed=false unless value_bboxes_available=true and bbox_verification_level=value",
            "owner_phase": "Phase7S",
            "notes": "Current value_bboxes_available=false keeps value-level claims disabled.",
        },
        {
            "gate_id": "binding_review_at_least_reviewed",
            "gate_name": "Binding review at least reviewed",
            "required_for_promotion": "true",
            "current_preview_status": "fail",
            "blocks_current_preview_units": "true",
            "block_reason": "binding_warning_level_only",
            "required_evidence": "binding_review_level is reviewed or verified; warning is not promotable",
            "owner_phase": "Phase7S",
            "notes": "Warning-level binding blocks production-ready citation.",
        },
        {
            "gate_id": "source_span_granularity_explicit",
            "gate_name": "Source span granularity explicit",
            "required_for_promotion": "true",
            "current_preview_status": "needs_review",
            "blocks_current_preview_units": "true",
            "block_reason": "source_span_granularity_not_verified",
            "required_evidence": "source_span_granularity is explicit and compatible with citation_scope",
            "owner_phase": "Phase7S",
            "notes": "Granularity must be auditable before support selection or citation.",
        },
        {
            "gate_id": "production_ready_true_independent_gate",
            "gate_name": "production_ready=true only after independent gate",
            "required_for_promotion": "true",
            "current_preview_status": "fail",
            "blocks_current_preview_units": "true",
            "block_reason": "production_ready_false",
            "required_evidence": "table_index_quality_gate_status=pass from independent quality gate, not retrieval_ready",
            "owner_phase": "Phase7S",
            "notes": "production_ready=true cannot be inferred from retrieval readiness or reranker score.",
        },
        {
            "gate_id": "index_unit_status_not_preview_only",
            "gate_name": "Index unit status is not preview_only",
            "required_for_promotion": "true",
            "current_preview_status": "fail",
            "blocks_current_preview_units": "true",
            "block_reason": "index_unit_status_preview_only",
            "required_evidence": "index_unit_status is production_candidate or production_ready, never preview_only",
            "owner_phase": "Phase7S",
            "notes": "This gate explicitly blocks current preview table units.",
        },
        {
            "gate_id": "non_table_query_guard_enforced",
            "gate_name": "Non-table query guard enforced",
            "required_for_promotion": "true",
            "current_preview_status": "needs_review",
            "blocks_current_preview_units": "true",
            "block_reason": "non_table_query_guard_missing",
            "required_evidence": "non_table_query blocks table branch/support/citation even if reranker score is high",
            "owner_phase": "Phase7S",
            "notes": "Reranker score cannot bypass query-type policy.",
        },
        {
            "gate_id": "rollback_metadata_present",
            "gate_name": "Rollback metadata present",
            "required_for_promotion": "true",
            "current_preview_status": "fail",
            "blocks_current_preview_units": "true",
            "block_reason": "rollback_manifest_missing",
            "required_evidence": "rollback_manifest_id and previous active build pointer are present and checksummed",
            "owner_phase": "Phase7S",
            "notes": "No production pointer should move without rollback metadata.",
        },
        {
            "gate_id": "typed_citation_schema_available",
            "gate_name": "Typed citation schema available",
            "required_for_promotion": "true",
            "current_preview_status": "fail",
            "blocks_current_preview_units": "true",
            "block_reason": "typed_schema_not_integrated",
            "required_evidence": "TableEvidenceCitation schema is versioned and integrated into production binder policy",
            "owner_phase": "Future production binder phase",
            "notes": "Phase7Q schema prototype alone is not production citation readiness.",
        },
        {
            "gate_id": "metadata_contract_valid",
            "gate_name": "Metadata contract valid",
            "required_for_promotion": "true",
            "current_preview_status": "fail",
            "blocks_current_preview_units": "true",
            "block_reason": "production_metadata_contract_incomplete",
            "required_evidence": "build/version/approval/citation/provenance/confidence/bbox/binding fields satisfy production metadata contract",
            "owner_phase": "Phase7S",
            "notes": "Current metadata is sandbox-compatible but not production-complete.",
        },
        {
            "gate_id": "checksum_build_manifest_valid",
            "gate_name": "Checksum and build manifest valid",
            "required_for_promotion": "true",
            "current_preview_status": "fail",
            "blocks_current_preview_units": "true",
            "block_reason": "checksum_or_build_manifest_missing",
            "required_evidence": "checksum_manifest_id covers every production artifact and matches build metadata",
            "owner_phase": "Phase7S",
            "notes": "Promotion requires immutable artifact identity.",
        },
    ]


def rollback_rows() -> list[dict[str, str]]:
    base_false = {
        "table_branch_executed_after_rollback": "false",
        "table_support_selected_after_rollback": "false",
        "table_citation_emitted_after_rollback": "false",
        "answer_visible_table_evidence_after_rollback": "false",
        "normal_only_restored": "true",
    }
    return [
        {
            "scenario_id": "flag_disabled",
            "trigger": "table_index_retrieval_enabled=false",
            "detection_gate": "branch_gate",
            "rollback_action": "Keep table branch disabled and preserve normal-only path.",
            "active_build_pointer_state": "unchanged_or_disabled",
            "notes": "Default safe state.",
            **base_false,
        },
        {
            "scenario_id": "table_index_unavailable",
            "trigger": "candidate table index cannot be opened",
            "detection_gate": "index_availability_gate",
            "rollback_action": "Do not execute table retrieval; fall back to normal-only.",
            "active_build_pointer_state": "previous_active_or_disabled",
            "notes": "Answer path must not fail because the table index is unavailable.",
            **base_false,
        },
        {
            "scenario_id": "table_index_schema_mismatch",
            "trigger": "table_index_unit_schema_version mismatch",
            "detection_gate": "schema_gate",
            "rollback_action": "Reject candidate build before merge.",
            "active_build_pointer_state": "previous_active_or_disabled",
            "notes": "Mismatched schema cannot feed rerank or support.",
            **base_false,
        },
        {
            "scenario_id": "canonical_source_manifest_missing",
            "trigger": "canonical_source_manifest_id missing or unresolved",
            "detection_gate": "canonical_source_gate",
            "rollback_action": "Block formal citation and keep normal-only path.",
            "active_build_pointer_state": "previous_active_or_disabled",
            "notes": "Debug paths cannot substitute for canonical paper source.",
            **base_false,
        },
        {
            "scenario_id": "metadata_contract_fail",
            "trigger": "required production metadata missing or invalid",
            "detection_gate": "metadata_contract_gate",
            "rollback_action": "Drop table candidates and restore normal-only candidate pool.",
            "active_build_pointer_state": "previous_active_or_disabled",
            "notes": "Sandbox metadata is insufficient for production support.",
            **base_false,
        },
        {
            "scenario_id": "citation_guard_fail",
            "trigger": "formal/debug source split, scope, or value-claim guard fails",
            "detection_gate": "citation_guard",
            "rollback_action": "Remove table support/citation candidates and restore normal-only citation path.",
            "active_build_pointer_state": "previous_active_or_disabled",
            "notes": "CSV/crop formal source misuse triggers rollback.",
            **base_false,
        },
        {
            "scenario_id": "production_ready_guard_fail",
            "trigger": "production_ready=false or no independent quality gate pass",
            "detection_gate": "production_ready_gate",
            "rollback_action": "Reject table support and citation.",
            "active_build_pointer_state": "previous_active_or_disabled",
            "notes": "retrieval_ready cannot substitute for production_ready.",
            **base_false,
        },
        {
            "scenario_id": "preview_only_guard_fail",
            "trigger": "index_unit_status=preview_only",
            "detection_gate": "preview_guard",
            "rollback_action": "Reject preview units from production candidate path.",
            "active_build_pointer_state": "previous_active_or_disabled",
            "notes": "Current preview table units are blocked.",
            **base_false,
        },
        {
            "scenario_id": "reranker_high_score_bypass_attempt",
            "trigger": "table candidate scores high while policy/metadata/citation gate fails",
            "detection_gate": "safety_gate_after_rerank",
            "rollback_action": "Ignore reranker score as safety signal and remove blocked table candidates.",
            "active_build_pointer_state": "previous_active_or_disabled",
            "notes": "Reranker score is ranking evidence only, not production safety evidence.",
            **base_false,
        },
        {
            "scenario_id": "active_build_pointer_rollback",
            "trigger": "post-promote monitoring or validation failure",
            "detection_gate": "active_pointer_guard",
            "rollback_action": "Restore previous active build pointer from rollback_manifest_id.",
            "active_build_pointer_state": "restored_previous_build",
            "notes": "Pointer rollback must be atomic and auditable.",
            **base_false,
        },
        {
            "scenario_id": "hard_disable_to_normal_only",
            "trigger": "operator hard disable or emergency guardrail trip",
            "detection_gate": "hard_disable_gate",
            "rollback_action": "Set table_index_retrieval_enabled=false and disable table support/citation/answer visibility.",
            "active_build_pointer_state": "disabled",
            "notes": "Table branch not executed; table support not selected; table citation not emitted; normal-only path restored.",
            **base_false,
        },
    ]


def rollout_rows() -> list[dict[str, str]]:
    return [
        {
            "stage": "disabled",
            "table_index_readable": "false",
            "table_branch_executes": "false",
            "table_candidates_enter_rerank_input": "false",
            "table_evidence_enters_support_pack": "false",
            "table_citation_allowed": "false",
            "answer_visible": "false",
            "rollback_condition": "Any uncertainty or default production state.",
            "required_gate": "none; hard disabled",
            "phase7r_execution_allowed": "true",
            "default_state": "allowed_design_state",
        },
        {
            "stage": "shadow_index_build",
            "table_index_readable": "false",
            "table_branch_executes": "false",
            "table_candidates_enter_rerank_input": "false",
            "table_evidence_enters_support_pack": "false",
            "table_citation_allowed": "false",
            "answer_visible": "false",
            "rollback_condition": "Build manifest/schema/checksum/canonical source gate missing.",
            "required_gate": "artifact manifest and independent quality gate design only",
            "phase7r_execution_allowed": "true",
            "default_state": "proposal_only",
        },
        {
            "stage": "shadow_retrieval_debug",
            "table_index_readable": "true",
            "table_branch_executes": "true",
            "table_candidates_enter_rerank_input": "false",
            "table_evidence_enters_support_pack": "false",
            "table_citation_allowed": "false",
            "answer_visible": "false",
            "rollback_condition": "Any metadata, canonical source, or query-type guard failure.",
            "required_gate": "shadow read gate; no user-visible effect",
            "phase7r_execution_allowed": "false",
            "default_state": "future_only",
        },
        {
            "stage": "active_merge_dry_run",
            "table_index_readable": "true",
            "table_branch_executes": "true",
            "table_candidates_enter_rerank_input": "true",
            "table_evidence_enters_support_pack": "false",
            "table_citation_allowed": "false",
            "answer_visible": "false",
            "rollback_condition": "Rerank compatibility, non-table guard, or safety gate failure.",
            "required_gate": "active merge dry-run approval",
            "phase7r_execution_allowed": "false",
            "default_state": "future_only",
        },
        {
            "stage": "support_pack_dry_run",
            "table_index_readable": "true",
            "table_branch_executes": "true",
            "table_candidates_enter_rerank_input": "true",
            "table_evidence_enters_support_pack": "true",
            "table_citation_allowed": "false",
            "answer_visible": "false",
            "rollback_condition": "Support selection or citation readiness gate failure.",
            "required_gate": "support dry-run approval; formal citation disabled",
            "phase7r_execution_allowed": "false",
            "default_state": "future_only",
        },
        {
            "stage": "canary_no_answer",
            "table_index_readable": "true",
            "table_branch_executes": "true",
            "table_candidates_enter_rerank_input": "true",
            "table_evidence_enters_support_pack": "true",
            "table_citation_allowed": "false",
            "answer_visible": "false",
            "rollback_condition": "Any production gate failure, monitoring anomaly, or rollback drill failure.",
            "required_gate": "all promotion gates except answer visibility",
            "phase7r_execution_allowed": "false",
            "default_state": "blocked",
        },
        {
            "stage": "canary_answer_gated",
            "table_index_readable": "true",
            "table_branch_executes": "true",
            "table_candidates_enter_rerank_input": "true",
            "table_evidence_enters_support_pack": "true",
            "table_citation_allowed": "true",
            "answer_visible": "true",
            "rollback_condition": "Citation, answer, monitoring, or rollback drill failure.",
            "required_gate": "production canary approval with typed citation schema and LLM answer smoke",
            "phase7r_execution_allowed": "false",
            "default_state": "blocked",
        },
        {
            "stage": "production",
            "table_index_readable": "true",
            "table_branch_executes": "true",
            "table_candidates_enter_rerank_input": "true",
            "table_evidence_enters_support_pack": "true",
            "table_citation_allowed": "true",
            "answer_visible": "true",
            "rollback_condition": "Any post-promotion quality, safety, pointer, or citation failure.",
            "required_gate": "full production promotion gate and rollback drill pass",
            "phase7r_execution_allowed": "false",
            "default_state": "blocked",
        },
    ]


def risk_rows() -> list[dict[str, str]]:
    return [
        {
            "risk_id": "preview_contamination",
            "risk": "preview contamination",
            "current_status": "active_blocker",
            "mitigation": "Promotion gate rejects index_unit_status=preview_only and requires independent production_ready=true.",
            "owner_phase": "Phase7S",
            "block_production": "true",
            "notes": "Current units are preview_only.",
        },
        {
            "risk_id": "canonical_source_missing",
            "risk": "canonical source missing",
            "current_status": "active_blocker",
            "mitigation": "Require canonical_source_manifest_id before build promotion or formal citation.",
            "owner_phase": "Phase7S",
            "block_production": "true",
            "notes": "Debug paths cannot substitute for paper source.",
        },
        {
            "risk_id": "csv_crop_formal_citation_misuse",
            "risk": "CSV/crop formal citation misuse",
            "current_status": "guard_required",
            "mitigation": "Keep CSV/crop/markdown paths in provenance_debug only and validate canonical_source.source_file.",
            "owner_phase": "Phase7S",
            "block_production": "true",
            "notes": "Phase7Q/Q-1 already prove this must remain a guard.",
        },
        {
            "risk_id": "table_scope_false_binding",
            "risk": "table scope false binding",
            "current_status": "active_blocker",
            "mitigation": "Require table id/caption/page validation and binding review >= reviewed.",
            "owner_phase": "Phase7S",
            "block_production": "true",
            "notes": "Warning-level binding is not production ready.",
        },
        {
            "risk_id": "row_cell_group_overclaim",
            "risk": "row/cell-group overclaim",
            "current_status": "active_blocker",
            "mitigation": "Require citation_scope and source_span_granularity to match reviewed row/header scope.",
            "owner_phase": "Phase7S",
            "block_production": "true",
            "notes": "Cell-group claims cannot broaden to whole-table claims.",
        },
        {
            "risk_id": "value_level_overclaim",
            "risk": "value-level overclaim",
            "current_status": "guard_required",
            "mitigation": "Forbid citation_scope=value and keep value_level_citation_claim_allowed=false unless value bboxes are verified.",
            "owner_phase": "Future bbox verification phase",
            "block_production": "true",
            "notes": "Current value_bboxes_available=false.",
        },
        {
            "risk_id": "reranker_high_score_bypass",
            "risk": "reranker high-score bypass",
            "current_status": "guard_required",
            "mitigation": "Treat reranker score as ranking only; safety gates run before support/citation/answer visibility.",
            "owner_phase": "Phase7S",
            "block_production": "true",
            "notes": "Phase7P observed high-score malformed/non-table table cases remain possible.",
        },
        {
            "risk_id": "rollback_incomplete",
            "risk": "rollback incomplete",
            "current_status": "guard_required",
            "mitigation": "Hard disable restores normal-only path and blocks branch/support/citation/answer-visible table evidence.",
            "owner_phase": "Phase7S",
            "block_production": "true",
            "notes": "Rollback drill is required before any canary.",
        },
        {
            "risk_id": "binder_integration_gap",
            "risk": "binder integration gap",
            "current_status": "active_blocker",
            "mitigation": "Require typed TableEvidenceCitation production binder integration before formal table citation.",
            "owner_phase": "Future production binder phase",
            "block_production": "true",
            "notes": "Q-1 mapper dry-run is not production binding.",
        },
        {
            "risk_id": "production_index_drift",
            "risk": "production index drift",
            "current_status": "design_gap",
            "mitigation": "Use immutable build id, source corpus snapshot id, checksum manifest, and active pointer rollback.",
            "owner_phase": "Phase7S",
            "block_production": "true",
            "notes": "Phase7R only designs the mechanism.",
        },
    ]


def manifest_template() -> dict[str, Any]:
    return {
        "manifest_kind": "production_table_index_artifact_manifest_template",
        "manifest_template_version": "phase7r-proposal-v1",
        "template_only": True,
        "production_artifact_created": False,
        "phase7r_scope": {
            "proposal_only": True,
            "production_index_built": False,
            "preview_units_upgraded": False,
            "formal_citation_enabled": False,
        },
        "build_metadata": {
            "table_index_version": "REQUIRED_FUTURE_VALUE",
            "table_index_build_id": "REQUIRED_IMMUTABLE_BUILD_ID",
            "table_index_unit_schema_version": "REQUIRED_FUTURE_VALUE",
            "source_corpus_snapshot_id": "REQUIRED_FUTURE_VALUE",
            "canonical_source_manifest_id": "REQUIRED_FUTURE_VALUE",
            "promotion_approval_id": "REQUIRED_FUTURE_VALUE",
            "rollback_manifest_id": "REQUIRED_FUTURE_VALUE",
            "table_index_quality_gate_status": "must_be_passed_by_independent_gate",
            "checksum_manifest_id": "REQUIRED_FUTURE_VALUE",
            "created_by_phase": "future_phase_not_phase7r",
        },
        "source_corpus_snapshot_metadata": {
            "source_corpus_snapshot_id": "REQUIRED_FUTURE_VALUE",
            "source_corpus_kind": "canonical_paper_corpus_snapshot",
            "source_corpus_checksum": "REQUIRED_FUTURE_VALUE",
            "bm25_or_milvus_read_required_for_phase7r": False,
        },
        "artifacts": {
            "production_table_units_jsonl": {
                "artifact_label": "production table units JSONL",
                "required": True,
                "path_template": "production/table_index/{table_index_build_id}/table_units.jsonl",
                "must_not_use_preview_only_units": True,
                "checksum_required": True,
            },
            "table_unit_schema_manifest": {
                "artifact_label": "table unit schema manifest",
                "required": True,
                "path_template": "production/table_index/{table_index_build_id}/table_unit_schema_manifest.json",
                "checksum_required": True,
            },
            "table_evidence_citation_schema_manifest": {
                "artifact_label": "TableEvidenceCitation schema manifest",
                "required": True,
                "path_template": "production/table_index/{table_index_build_id}/table_evidence_citation_schema_manifest.json",
                "checksum_required": True,
            },
            "canonical_source_manifest": {
                "artifact_label": "canonical source manifest",
                "required": True,
                "path_template": "production/table_index/{table_index_build_id}/canonical_source_manifest.json",
                "formal_source_authority": True,
                "checksum_required": True,
            },
            "debug_provenance_manifest": {
                "artifact_label": "debug provenance manifest",
                "required": True,
                "path_template": "production/table_index/{table_index_build_id}/debug_provenance_manifest.json",
                "debug_only_paths": ["source_csv_path", "source_pdf_crop_path", "source_markdown_path"],
                "checksum_required": True,
            },
            "validation_report": {
                "artifact_label": "validation report",
                "required": True,
                "path_template": "production/table_index/{table_index_build_id}/validation_report.md",
                "must_include_independent_quality_gate": True,
                "checksum_required": True,
            },
            "promotion_approval_record": {
                "artifact_label": "promotion approval record",
                "required": True,
                "path_template": "production/table_index/{table_index_build_id}/promotion_approval_record.json",
                "approval_required_before_active_pointer_move": True,
                "checksum_required": True,
            },
            "rollback_record": {
                "artifact_label": "rollback record",
                "required": True,
                "path_template": "production/table_index/{table_index_build_id}/rollback_record.json",
                "previous_active_build_pointer_required": True,
                "checksum_required": True,
            },
            "checksum_manifest": {
                "artifact_label": "checksum manifest",
                "required": True,
                "path_template": "production/table_index/{table_index_build_id}/checksum_manifest.json",
                "covers_all_artifacts": True,
            },
        },
        "formal_source_policy": {
            "canonical_source_manifest_required": True,
            "csv_crop_markdown_paths_debug_only": True,
            "production_build_must_not_treat_csv_or_crop_as_formal_source": True,
        },
        "promotion_policy": {
            "production_ready_true_requires_independent_quality_gate": True,
            "retrieval_ready_cannot_imply_production_ready": True,
            "reranker_score_cannot_be_safety_signal": True,
            "preview_only_units_blocked": True,
        },
    }


def render_guardrail() -> str:
    return """# Phase7R Guardrail Freeze

Phase7R is a proposal phase, not a production implementation.

Frozen constraints:

- Do not upgrade preview units.
- Do not enable formal table citation.
- Do not build a production table index.
- Do not run retrieval or generation.
- Do not access Milvus.
- Do not read or query official BM25.
- Do not modify `src/`.
- Do not modify `configs/`.
- Do not modify ingestion pipeline.
- Do not run embedding.
- Do not run reranker.
- Do not call Qwen, LLM, RAGAS, OCR, or VLM.
- Do not generate answers.
- Do not generate formal production citations.
- Do not enter Route C implementation.

Route C remains backlog. Phase7R may only create proposal reports, matrices, a manifest template, a validator, and tests."""


def render_blocker_review() -> str:
    rows = [
        ("canonical paper source unresolved", "active", "Blocks formal source identity and production citation."),
        ("table units still preview_only", "active", "Blocks production promotion."),
        ("production_ready=false", "active", "Blocks production support and formal citation."),
        ("value_bboxes_available=false", "active", "Blocks value-level citation claims."),
        ("binding warning-level only", "active", "Blocks production-ready table/row/cell-group binding."),
        ("typed schema not integrated into production binder", "active", "Schema prototype is not production binding."),
        ("mapper dry-run not equal production binding", "active", "Q-1 proves mapping shape only."),
        ("reranker score cannot be safety signal", "active", "High rank cannot bypass metadata/citation gates."),
        ("formal citation loop not closed", "active", "Citation schema, binder, source resolution, and answer visibility are not closed."),
        ("production table index build/promote/rollback not designed", "being_designed_in_phase7r", "This phase proposes the design but does not execute it."),
    ]
    table = markdown_table(
        [
            {"blocker": blocker, "status": status, "production_impact": impact}
            for blocker, status, impact in rows
        ],
        ["blocker", "status", "production_impact"],
    )
    return f"""# Phase7R Current Blocker Review

Inputs reviewed: Phase7Q schema prototype, Phase7Q-1 mapper dry-run, Phase7N wiring design, Phase7O dry-run, Phase7P reranker smoke, and Phase7M contract hardening.

{table}

Conclusion: current state should not actually build a production table index. The table artifacts remain preview/debug assets until canonical source resolution, independent production readiness gates, typed binder integration, promotion metadata, and rollback metadata all pass."""


def render_build_proposal() -> str:
    fields = [
        "table_index_version",
        "table_index_build_id",
        "table_index_unit_schema_version",
        "source_corpus_snapshot_id",
        "canonical_source_manifest_id",
        "promotion_approval_id",
        "rollback_manifest_id",
        "table_index_quality_gate_status",
        "checksum_manifest_id",
    ]
    field_rows = [
        {
            "field": field,
            "purpose": {
                "table_index_version": "Stable semantic production table index version.",
                "table_index_build_id": "Immutable build identity used for promotion and rollback.",
                "table_index_unit_schema_version": "Schema version for production table units.",
                "source_corpus_snapshot_id": "Immutable canonical corpus snapshot identity.",
                "canonical_source_manifest_id": "Formal paper/table/page source resolution manifest.",
                "promotion_approval_id": "Approval artifact authorizing active pointer movement.",
                "rollback_manifest_id": "Record of previous pointer and hard-disable fallback.",
                "table_index_quality_gate_status": "Independent quality gate result, not inferred from retrieval.",
                "checksum_manifest_id": "Checksum coverage for every production artifact.",
            }[field],
        }
        for field in fields
    ]
    return f"""# Production Table Index Build Proposal

Phase7R defines the future build shape only. It does not build a production table index.

## Future Build Inputs

- Production table unit candidates from a future extractor/indexing path, not direct reuse of current `preview_only` units.
- Canonical paper source manifest resolving `doc_id`, paper title, canonical source file, DOI/PMID when available, table id, caption, and page scope.
- Table unit schema manifest.
- TableEvidenceCitation schema manifest.
- Debug provenance manifest for CSV/crop/markdown paths.
- Independent table index quality gate report.
- Source corpus snapshot metadata.

Production table units cannot directly use `preview_only` units. They must be rebuilt or promoted through an independent quality gate that can reject current preview artifacts.

## Future Build Outputs

- Production table units JSONL.
- Versioned table unit schema manifest.
- Versioned TableEvidenceCitation schema manifest.
- Canonical source manifest.
- Debug provenance manifest.
- Validation report.
- Promotion approval record.
- Rollback record.
- Checksum manifest.
- Build metadata.
- Source corpus snapshot metadata.

## Required Version And Control Fields

{markdown_table(field_rows, ["field", "purpose"])}

## Required Production Rules

- Production build input must pass an independent quality gate before any active pointer move.
- `production_ready=true` must come from that independent gate and cannot be derived from `retrieval_ready`, retrieval presence, rank, or reranker score.
- Production build must not treat CSV/crop/markdown paths as formal citation sources.
- Formal citation source must come from the canonical source manifest.
- Current Phase7Q/Q-1 artifacts remain debug/prototype evidence only."""


def render_manifest_report(template: dict[str, Any]) -> str:
    artifact_rows = [
        {
            "artifact": artifact["artifact_label"],
            "required": str(artifact.get("required", "")).lower(),
            "template_path": artifact.get("path_template", ""),
        }
        for artifact in template["artifacts"].values()
    ]
    return f"""# Production Index Artifact Manifest

The JSON template is stored at:

- `data/experiments/v7_phase7_table_index_production_proposal/production_index_artifact_manifest_template.json`

This round only generates a manifest template. It does not generate real production artifacts and does not build a production table index.

## Required Manifest Sections

{markdown_table(artifact_rows, ["artifact", "required", "template_path"])}

The template also contains `build_metadata` and `source_corpus_snapshot_metadata`. The build metadata requires `table_index_version`, `table_index_build_id`, `table_index_unit_schema_version`, `source_corpus_snapshot_id`, `canonical_source_manifest_id`, `promotion_approval_id`, `rollback_manifest_id`, `table_index_quality_gate_status`, and `checksum_manifest_id`.

CSV/crop/markdown paths are only allowed in the debug provenance manifest. They are not formal citation sources."""


def render_promotion_gate_report(rows: list[dict[str, str]]) -> str:
    table = markdown_table(
        rows,
        [
            "gate_id",
            "current_preview_status",
            "blocks_current_preview_units",
            "block_reason",
        ],
    )
    return f"""# Promotion Gate Matrix

Every production candidate must pass the gate matrix before promotion.

{table}

The matrix intentionally blocks current preview table units through canonical source, binding review, production readiness, preview-only, rollback metadata, typed schema, metadata contract, and checksum/build manifest gates.

Reranker score is not a safety signal and cannot bypass any gate."""


def render_rollback_design(rows: list[dict[str, str]]) -> str:
    table = markdown_table(
        rows,
        [
            "scenario_id",
            "detection_gate",
            "active_build_pointer_state",
            "normal_only_restored",
        ],
    )
    return f"""# Promotion / Rollback Design

## Promotion States

1. `disabled`: table branch is not executed and normal-only path is active.
2. `shadow_candidate_build`: candidate artifacts may be assembled in a future phase, but runtime cannot read them.
3. `candidate_active_dry_run`: table branch may be evaluated without answer-visible effects.
4. `promote_manifest`: promotion approval, checksum, canonical source, and rollback records are complete.
5. `active_build_pointer`: active pointer moves only after all promotion gates pass.
6. `rollback_to_previous_build`: restore previous active build pointer from rollback manifest.
7. `hard_disable_to_normal_only_path`: set table branch disabled and restore normal-only behavior.

Rollback guarantees after every scenario:

- table branch not executed;
- table support not selected;
- table citation not emitted;
- answer-visible table evidence absent;
- active build pointer restored or disabled;
- normal-only path restored.

## Scenario Matrix

{table}"""


def render_citation_readiness() -> str:
    return """# Citation Readiness Coupling

Phase7Q schema prototype pass does not equal production citation ready.

Production gates must enforce:

- no typed schema, no formal table citation;
- no canonical source, no formal table citation;
- preview_only blocks formal citation;
- production_ready=false blocks formal citation;
- mapper dry-run is not production binder;
- CSV/crop stay debug-only and must not enter canonical_source or public Citation.source_file;
- citation_scope=value is forbidden;
- value-level citation disabled unless value bbox verified;
- binding warning-level blocks production-ready citation.

Coupling rule: formal table citation may be enabled only when the production table index candidate, typed TableEvidenceCitation schema, canonical source manifest, metadata contract, binding review, and rollback manifest all pass the promotion gate. Any failure keeps table evidence debug-only or disables the table branch."""


def render_rollout_plan(rows: list[dict[str, str]]) -> str:
    table = markdown_table(
        rows,
        [
            "stage",
            "table_index_readable",
            "table_branch_executes",
            "table_candidates_enter_rerank_input",
            "table_evidence_enters_support_pack",
            "table_citation_allowed",
            "answer_visible",
            "phase7r_execution_allowed",
            "default_state",
        ],
    )
    return f"""# Canary / Shadow Rollout Plan

Phase7R only designs rollout stages. It does not enter `canary_no_answer`, `canary_answer_gated`, or `production` execution.

{table}

Any stage at or beyond `shadow_retrieval_debug` requires future authorization and matching gates. `production` is blocked by default."""


def render_risk_register(rows: list[dict[str, str]]) -> str:
    table = markdown_table(
        rows,
        ["risk_id", "risk", "current_status", "mitigation", "owner_phase", "block_production"],
    )
    return f"""# Risk Register

{table}

All listed risks block production until their mitigation is verified by a future phase."""


def render_validation_placeholder() -> str:
    return """# Phase7R Validation Report

- validation_status: `pending`
- output: `results/v7_phase7_table_index_production_proposal/phase7r_validation_results.csv`

Run `python scripts/evaluation/phase7r_validate_table_index_production_proposal.py` to validate Phase7R proposal artifacts."""


def render_summary(validation_status: str = "pending") -> str:
    report_files = "\n".join(
        f"- `reports/v7_phase7_table_index_production_proposal/{path}`" for path in REPORT_PATHS
    )
    data_files = "\n".join(
        f"- `data/experiments/v7_phase7_table_index_production_proposal/{path}`"
        for path in STRUCTURED_PATHS
    )
    result_files = "\n".join(
        f"- `results/v7_phase7_table_index_production_proposal/{path}`" for path in RESULT_PATHS
    )
    return f"""# Phase7R Summary

## 1. Generated Files

Reports:

{report_files}

Structured files:

{data_files}

Results:

{result_files}

Scripts/tests:

- `scripts/evaluation/phase7r_build_table_index_production_proposal.py`
- `scripts/evaluation/phase7r_validate_table_index_production_proposal.py`
- `tests/test_phase7r_table_index_production_proposal.py`

## 2. Guardrail Status

- Modified `src/`: no.
- Modified `configs/`: no.
- Accessed Milvus / official BM25: no.
- Ran embedding / reranker / LLM: no.
- Built production table index: no.
- Generated answer: no.
- Generated formal production citation: no.
- Entered Route C implementation: no.

## 3. Conclusions

- Blocker review: current state should not actually build a production table index.
- Production table index build proposal: future build requires canonical source, independent quality gate, manifest/checksum/version fields, and no direct reuse of preview_only units.
- Artifact manifest template: template only; no real production artifact generated.
- Promotion gate matrix: blocks current preview table units.
- Promotion / rollback design: rollback can restore normal-only path or hard-disable table branch.
- Citation readiness coupling: Phase7Q schema prototype pass does not equal production citation ready.
- Canary / shadow rollout plan: production and canary stages are blocked by default in Phase7R.
- Risk register: all listed risks block production until future mitigation.
- Validation result: `{validation_status}`.

## 4. Recommendations

- Recommend Phase7S: yes, Canonical Source Resolution + Production Readiness Gate Dry-Run.
- Recommend production: no.
- Recommend directly building production index: no.
- Recommend extractor rework: no.
- Recommend continued large manual annotation: no.
- Route C remains backlog: yes."""


def main() -> None:
    ensure_dirs()

    gates = promotion_gate_rows()
    rollbacks = rollback_rows()
    rollouts = rollout_rows()
    risks = risk_rows()
    template = manifest_template()

    write_text(REPORT_DIR / "phase7r_guardrail.md", render_guardrail())
    write_text(REPORT_DIR / "phase7r_blocker_review.md", render_blocker_review())
    write_text(REPORT_DIR / "production_table_index_build_proposal.md", render_build_proposal())
    write_text(REPORT_DIR / "production_index_artifact_manifest.md", render_manifest_report(template))
    write_text(REPORT_DIR / "promotion_gate_matrix.md", render_promotion_gate_report(gates))
    write_text(REPORT_DIR / "promotion_rollback_design.md", render_rollback_design(rollbacks))
    write_text(REPORT_DIR / "citation_readiness_coupling.md", render_citation_readiness())
    write_text(REPORT_DIR / "canary_shadow_rollout_plan.md", render_rollout_plan(rollouts))
    write_text(REPORT_DIR / "risk_register.md", render_risk_register(risks))
    write_text(REPORT_DIR / "phase7r_validation_report.md", render_validation_placeholder())
    write_text(REPORT_DIR / "phase7r_summary.md", render_summary())

    write_json(DATA_DIR / "production_index_artifact_manifest_template.json", template)
    write_csv(
        DATA_DIR / "promotion_gate_matrix.csv",
        gates,
        [
            "gate_id",
            "gate_name",
            "required_for_promotion",
            "current_preview_status",
            "blocks_current_preview_units",
            "block_reason",
            "required_evidence",
            "owner_phase",
            "notes",
        ],
    )
    write_csv(
        DATA_DIR / "rollback_scenario_matrix.csv",
        rollbacks,
        [
            "scenario_id",
            "trigger",
            "detection_gate",
            "rollback_action",
            "table_branch_executed_after_rollback",
            "table_support_selected_after_rollback",
            "table_citation_emitted_after_rollback",
            "answer_visible_table_evidence_after_rollback",
            "active_build_pointer_state",
            "normal_only_restored",
            "notes",
        ],
    )
    write_csv(
        DATA_DIR / "rollout_stage_matrix.csv",
        rollouts,
        [
            "stage",
            "table_index_readable",
            "table_branch_executes",
            "table_candidates_enter_rerank_input",
            "table_evidence_enters_support_pack",
            "table_citation_allowed",
            "answer_visible",
            "rollback_condition",
            "required_gate",
            "phase7r_execution_allowed",
            "default_state",
        ],
    )
    write_csv(
        DATA_DIR / "risk_register.csv",
        risks,
        [
            "risk_id",
            "risk",
            "current_status",
            "mitigation",
            "owner_phase",
            "block_production",
            "notes",
        ],
    )


if __name__ == "__main__":
    main()
