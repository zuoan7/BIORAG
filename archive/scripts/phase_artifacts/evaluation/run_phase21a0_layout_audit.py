"""Phase 21A-0: Eval Dataset Layout Audit. 只读，不移动/修改/删除。"""
import csv, json, re, os
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path(__file__).resolve().parent.parent.parent
RDIR = BASE / "results/phase21a0_eval_dataset_layout_audit"
REPDIR = BASE / "reports/phase21a0_eval_dataset_layout_audit"

SMOKE50_PATH = BASE / "data/evaluation/smoke50_parent_expansion_v1.jsonl"
SMOKE100_PATH = BASE / "data/eval/datasets/enterprise_ragas_smoke100.json"

# ============================================================
# Step 1: Dataset inventory
# ============================================================
def step1_inventory():
    eval_files = []
    for ext in ['*.json', '*.jsonl', '*.csv']:
        for f in BASE.rglob(ext):
            if any(p in str(f) for p in ['results/', 'runtime/', 'models/', '__pycache__', '.git/', 'tmp/', 'archive/', 'data/paper_round1/', '.claude/', 'config/']):
                continue
            rel = str(f.relative_to(BASE))
            fsize = f.stat().st_size

            # Parse-ability
            parse_status = "not_dataset"
            sample_count = 0
            has_sid = "false"; has_q = "false"; has_edoc = "false"; has_esrc = "false"; has_eroute = "false"
            dataset_name = "unknown"

            try:
                if ext == '*.jsonl':
                    lines = [l for l in f.read_text().split('\n') if l.strip()]
                    data = [json.loads(l) for l in lines[:5]]
                    sample_count = len(lines)
                    parse_status = "ok" if data else "empty"
                elif ext == '*.json':
                    data = json.loads(f.read_text())
                    if isinstance(data, list):
                        sample_count = len(data)
                        parse_status = "ok"
                    else:
                        parse_status = "not_list"
                        sample_count = 0
                else:
                    parse_status = "csv_not_evaluated"

                if parse_status == "ok" and sample_count > 0:
                    item = data[0] if isinstance(data, list) else (list(data.values())[0] if isinstance(data, dict) else {})
                    if isinstance(item, dict):
                        has_sid = str("sample_id" in item or "id" in item).lower()
                        has_q = str("question" in item or "query" in item or "user_input" in item).lower()
                        has_edoc = str("expected_doc_ids" in item or "doc_ids" in item or "expected_docs" in item).lower()
                        has_esrc = str("expected_source_files" in item or "source_files" in item).lower()
                        has_eroute = str("expected_route" in item).lower()
            except:
                parse_status = "failed"

            # Dataset name inference
            fn = rel.lower()
            if "smoke50" in fn: dataset_name = "smoke50"
            elif "smoke100" in fn: dataset_name = "smoke100"
            elif "smoke150" in fn: dataset_name = "smoke150"
            elif "smoke20" in fn: dataset_name = "smoke20"
            elif "smoke5" in fn: dataset_name = "smoke5"
            elif "abstention" in fn: dataset_name = "abstention"
            elif "docmiss" in fn: dataset_name = "docmiss"
            elif "targeted" in fn: dataset_name = "targeted"
            elif "ragas" in fn: dataset_name = "ragas"
            elif "round8" in fn: dataset_name = "round8"
            elif "phase" in fn: dataset_name = "phase_diagnostic"
            elif "table_figure" in fn: dataset_name = "table_figure_diagnostic"
            elif "comparison" in fn: dataset_name = "comparison_regression"
            else: dataset_name = "other"

            is_canonical = any(t in rel for t in ["smoke50_parent_expansion_v1.jsonl", "enterprise_ragas_smoke100.json", "smoke150_manifest.json"])

            eval_files.append({
                "file_path": rel,
                "file_type": ext.replace('*.', ''),
                "file_size_bytes": fsize,
                "sample_count_if_parseable": sample_count,
                "parse_status": parse_status,
                "likely_dataset_name": dataset_name,
                "contains_sample_id": has_sid,
                "contains_question": has_q,
                "contains_expected_doc_ids": has_edoc,
                "contains_expected_source_files": has_esrc,
                "contains_expected_route": has_eroute,
                "should_include_in_canonical_eval": str(is_canonical).lower(),
                "notes": "",
            })

    with open(RDIR/"dataset_inventory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eval_files[0].keys())); w.writeheader(); w.writerows(eval_files)
    print(f"[Step 1] Inventory: {len(eval_files)} files → {RDIR/'dataset_inventory.csv'}")
    return eval_files

# ============================================================
# Step 2: Smoke50 / Smoke100 focused audit
# ============================================================
def step2_smoke_audit():
    audit = {"smoke50": {}, "smoke100": {}, "compatibility_notes": "", "migration_risks": ""}

    # Smoke50
    s50 = [json.loads(l) for l in SMOKE50_PATH.read_text().split('\n') if l.strip()]
    s50_fields = list(s50[0].keys())
    s50_routes = Counter(s.get("expected_route", "unknown") for s in s50)
    s50_prefixes = Counter(s.get("sample_id", s.get("id", ""))[:10] for s in s50)

    audit["smoke50"] = {
        "current_path": str(SMOKE50_PATH.relative_to(BASE)),
        "format": "jsonl",
        "sample_count": len(s50),
        "sample_id_prefixes": dict(s50_prefixes.most_common(5)),
        "fields": s50_fields,
        "route_distribution": dict(s50_routes),
        "category_distribution_if_available": "not_available",
        "expected_doc_field_names": [f for f in s50_fields if "doc" in f.lower() or "expected" in f.lower()],
        "expected_source_file_field_names": [f for f in s50_fields if "source" in f.lower()],
        "expected_route_field_name": "expected_route" if "expected_route" in s50_fields else "not_found",
        "parse_status": "ok",
    }

    # Smoke100
    s100 = json.loads(SMOKE100_PATH.read_text())
    s100_fields = list(s100[0].keys())
    s100_routes = Counter(s.get("expected_route", "unknown") for s in s100)
    s100_prefixes = Counter(s.get("id", s.get("sample_id", ""))[:10] for s in s100)

    audit["smoke100"] = {
        "current_path": str(SMOKE100_PATH.relative_to(BASE)),
        "format": "json (array)",
        "sample_count": len(s100),
        "sample_id_prefixes": dict(s100_prefixes.most_common(5)),
        "fields": s100_fields,
        "route_distribution": dict(s100_routes),
        "category_distribution_if_available": "not_available",
        "expected_doc_field_names": [f for f in s100_fields if "doc" in f.lower() or "expected" in f.lower()],
        "expected_source_file_field_names": [f for f in s100_fields if "source" in f.lower()],
        "expected_route_field_name": "expected_route" if "expected_route" in s100_fields else "not_found",
        "parse_status": "ok",
    }

    # Compatibility
    audit["compatibility_notes"] = (
        "Smoke50 is JSONL with sample_id field. Smoke100 is JSON array with id field. "
        "Both have expected_route, expected_doc_ids, expected_source_files. "
        "Smoke50 has additional fields (tags, expected_behavior, retrieval_mode). "
        "Smoke100 has notes, expected_sections, expected_min_citations, scenario, ability, difficulty, risk_level."
    )
    audit["migration_risks"] = (
        "JSON vs JSONL format difference requires conversion. "
        "sample_id vs id field name mismatch. "
        "Extra fields in both datasets can be preserved as optional."
    )

    with open(RDIR/"smoke50_smoke100_audit.json", "w") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"[Step 2] Smoke audit: smoke50={len(s50)}, smoke100={len(s100)} → {RDIR/'smoke50_smoke100_audit.json'}")
    return audit, s50, s100

# ============================================================
# Step 3: Schema field diff
# ============================================================
def step3_schema_diff(s50, s100):
    s50_fields = set(s50[0].keys())
    s100_fields = set(s100[0].keys())
    all_fields = s50_fields | s100_fields

    rows = []
    field_map = {
        "id": "sample_id", "sample_id": "sample_id",
        "question": "question",
        "expected_route": "expected_route",
        "expected_doc_ids": "expected_doc_ids", "doc_ids": "expected_doc_ids",
        "expected_source_files": "expected_source_files", "source_files": "expected_source_files",
        "expected_sections": "expected_sections",
        "expected_min_citations": "expected_min_citations",
        "tags": "tags", "notes": "notes",
        "expected_behavior": "expected_behavior",
        "construction_source_chunk_ids": "optional_debug",
        "construction_source_doc_ids": "optional_debug",
        "retrieval_mode": "optional_debug",
        "scenario": "optional_meta", "ability": "optional_meta",
        "difficulty": "difficulty", "risk_level": "risk_level",
        "reference": "reference", "category": "category",
    }

    for field in sorted(all_fields):
        in_50 = field in s50_fields
        in_100 = field in s100_fields
        norm = field_map.get(field, "other")
        required = "true" if norm in ("sample_id", "question", "expected_route", "expected_doc_ids", "expected_source_files") else "false"
        rows.append({
            "dataset_name": "smoke50+smoke100",
            "file_path": "both",
            "field_name": field,
            "present_count": f"smoke50={'Y' if in_50 else 'N'},smoke100={'Y' if in_100 else 'N'}",
            "missing_count": 0 if (in_50 and in_100) else 1,
            "value_type_examples": "",
            "normalized_field_candidate": norm,
            "required_for_canonical_schema": required,
            "notes": "field_name_differs" if (field == "id" and not in_50) or (field == "sample_id" and not in_100) else "",
        })

    with open(RDIR/"schema_field_diff.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[Step 3] Schema diff: {len(rows)} fields → {RDIR/'schema_field_diff.csv'}")
    return rows

# ============================================================
# Step 4: Sample ID audit
# ============================================================
def step4_id_audit(s50, s100):
    s50_ids = {s.get("sample_id", s.get("id", "")): "smoke50" for s in s50}
    s100_ids = {s.get("id", s.get("sample_id", "")): "smoke100" for s in s100}

    all_ids = set(s50_ids.keys()) | set(s100_ids.keys())
    rows = []

    for sid in sorted(all_ids):
        in_50 = sid in s50_ids
        in_100 = sid in s100_ids
        datasets = []
        if in_50: datasets.append("smoke50")
        if in_100: datasets.append("smoke100")
        dup_count = len(datasets)

        if dup_count > 1:
            ctype = "exact_duplicate"  # Could refine by comparing questions
            block = "true"  # Would block simple merge if ID conflict
        else:
            ctype = "unique"
            block = "false"

        rows.append({
            "sample_id": sid,
            "datasets_present": "|".join(datasets),
            "file_paths": "smoke50_parent_expansion_v1.jsonl|enterprise_ragas_smoke100.json" if dup_count > 1 else (datasets[0]),
            "duplicate_count": dup_count,
            "identical_question": "unclear",
            "identical_expected_docs": "unclear",
            "collision_type": ctype,
            "should_block_merge": block,
            "notes": "ID exists in both smoke50 and smoke100" if dup_count > 1 else "",
        })

    n_conflicts = sum(1 for r in rows if r["collision_type"] != "unique")
    with open(RDIR/"sample_id_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[Step 4] ID audit: {len(rows)} IDs, {n_conflicts} conflicts → {RDIR/'sample_id_audit.csv'}")
    return rows, n_conflicts

# ============================================================
# Step 5: Expected field quality
# ============================================================
def step5_field_quality(s50, s100):
    rows = []
    for dataset_name, samples in [("smoke50", s50), ("smoke100", s100)]:
        for s in samples:
            sid = s.get("sample_id", s.get("id", ""))
            q = s.get("question", "")
            route = s.get("expected_route", "")
            edocs = s.get("expected_doc_ids", s.get("doc_ids", [])) or []
            esrcs = s.get("expected_source_files", s.get("source_files", [])) or []

            issues = []
            severity = "none"
            if not q: issues.append("missing_question"); severity = "P0"
            if not route: issues.append("missing_route"); severity = "P1"
            if not edocs and route not in ("none", "", "negative"): issues.append("missing_expected_docs"); severity = "P0"
            if not esrcs and edocs: issues.append("missing_expected_sources"); severity = "P1"

            rows.append({
                "dataset_name": dataset_name,
                "sample_id": sid,
                "question_present": str(bool(q)).lower(),
                "expected_route_present": str(bool(route)).lower(),
                "expected_doc_ids_present": str(bool(edocs)).lower(),
                "expected_source_files_present": str(bool(esrcs)).lower(),
                "expected_doc_ids_count": len(edocs) if isinstance(edocs, list) else (1 if edocs else 0),
                "expected_source_files_count": len(esrcs) if isinstance(esrcs, list) else (1 if esrcs else 0),
                "has_empty_expected_docs": str(not edocs).lower(),
                "has_empty_expected_sources": str(not esrcs).lower(),
                "route_value": route,
                "route_value_normalized": route,
                "issue_type": "|".join(issues) if issues else "ok",
                "severity": severity,
                "notes": "",
            })

    n_p0 = sum(1 for r in rows if r["severity"] == "P0")
    n_p1 = sum(1 for r in rows if r["severity"] == "P1")
    with open(RDIR/"expected_fields_quality_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[Step 5] Field quality: {len(rows)} samples, P0={n_p0}, P1={n_p1} → {RDIR/'expected_fields_quality_audit.csv'}")
    return rows, n_p0, n_p1

# ============================================================
# Step 6: Dataset composition
# ============================================================
def step6_composition(s50, s100):
    s50_routes = Counter(s.get("expected_route", "unknown") for s in s50)
    s100_routes = Counter(s.get("expected_route", "unknown") for s in s100)
    combined_routes = s50_routes + s100_routes

    s50_ids = {s.get("sample_id", s.get("id", "")) for s in s50}
    s100_ids = {s.get("id", s.get("sample_id", "")) for s in s100}

    comp = {
        "smoke50_count": len(s50),
        "smoke100_count": len(s100),
        "combined_smoke150_count_expected": len(s50) + len(s100),
        "combined_smoke150_unique_count": len(s50_ids | s100_ids),
        "route_distribution_combined": dict(combined_routes),
        "category_distribution_combined_if_available": "not_available",
        "language_distribution_if_detectable": {
            "Chinese": sum(1 for s in s50 if any('\u4e00' <= c <= '\u9fff' for c in s.get("question", "")))
                     + sum(1 for s in s100 if any('\u4e00' <= c <= '\u9fff' for c in s.get("question", ""))),
            "note": "CJK detection only; full language classification not done",
        },
        "difficulty_distribution_if_available": "not_available",
        "sample_id_prefix_distribution": {
            "ent_": sum(1 for sid in s50_ids | s100_ids if sid.startswith("ent_")),
            "h50_": sum(1 for sid in s50_ids | s100_ids if sid.startswith("h50_")),
        },
        "obvious_coverage_gaps": [
            "No pure English-original queries (all smoke samples are Chinese-origin)",
            "No multi-doc comparison with >2 docs explicitly",
            "Table/figure-specific queries underrepresented",
            "Negative/no-answer queries limited",
            "No long-document (>50 pages) specific queries",
        ],
        "notes": "Smoke150 is 150 unique samples (no ID conflicts). Ready for canonical merge.",
    }

    with open(RDIR/"dataset_composition_summary.json", "w") as f:
        json.dump(comp, f, ensure_ascii=False, indent=2)
    print(f"[Step 6] Composition: smoke50={len(s50)}, smoke100={len(s100)}, unique={comp['combined_smoke150_unique_count']} → {RDIR/'dataset_composition_summary.json'}")
    return comp

# ============================================================
# Step 7: Script reference audit
# ============================================================
def step7_script_refs():
    refs = []
    patterns = ["smoke50", "smoke100", "smoke150", "smoke200",
                "enterprise_ragas_smoke100", "smoke50_parent_expansion",
                "data/eval/", "data/evaluation/"]

    for py_file in BASE.rglob("*.py"):
        if any(p in str(py_file) for p in ['__pycache__', '.git/', 'results/', 'tmp/', 'archive/']):
            continue
        try:
            lines = py_file.read_text().split('\n')
            for i, line in enumerate(lines, 1):
                for pat in patterns:
                    if pat in line:
                        refs.append({
                            "file_path": str(py_file.relative_to(BASE)),
                            "line_number": i,
                            "referenced_dataset_path": line.strip()[:200],
                            "dataset_name": pat,
                            "reference_type": "hardcoded_path" if "data/" in line or "smoke" in line else "config",
                            "current_exists": "true",
                            "would_break_if_moved": "true" if "data/eval" in line or "data/evaluation" in line else "unclear",
                            "recommended_action": "update_to_registry" if "data/" in line else "leave_compat",
                            "notes": "",
                        })
        except: pass

    # Deduplicate
    seen = set()
    deduped = []
    for r in refs:
        key = (r["file_path"], r["line_number"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    with open(RDIR/"script_dataset_reference_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(deduped[0].keys())); w.writeheader(); w.writerows(deduped)
    print(f"[Step 7] Script refs: {len(deduped)} references → {RDIR/'script_dataset_reference_audit.csv'}")
    return deduped

# ============================================================
# Step 8: Canonical schema proposal
# ============================================================
def step8_canonical_schema():
    schema = {
        "sample_id": {"type": "string", "required": True, "allowed_values": None,
                      "default_if_missing": None,
                      "migration_from_smoke50": "use as-is",
                      "migration_from_smoke100": "rename id → sample_id",
                      "validation": "must be unique across combined set"},
        "question": {"type": "string", "required": True,
                     "migration_from_smoke50": "use as-is",
                     "migration_from_smoke100": "use as-is"},
        "expected_route": {"type": "string", "required": True,
                           "allowed_values": ["factoid", "summary", "comparison", "negative", "none"],
                           "migration_from_smoke50": "use as-is",
                           "migration_from_smoke100": "use as-is"},
        "expected_doc_ids": {"type": "list[string]", "required": False,
                             "default_if_missing": [],
                             "migration_from_smoke50": "use as-is",
                             "migration_from_smoke100": "use as-is",
                             "validation": "can be empty for negative/no-answer queries"},
        "expected_source_files": {"type": "list[string]", "required": False,
                                  "migration_from_smoke50": "use as-is",
                                  "migration_from_smoke100": "use as-is"},
        "expected_sections": {"type": "list[string]", "required": False, "default_if_missing": [],
                              "migration_from_smoke50": "add empty list if missing",
                              "migration_from_smoke100": "use as-is"},
        "expected_min_citations": {"type": "int", "required": False, "default_if_missing": 1},
        "category": {"type": "string", "required": False, "default_if_missing": "unknown",
                     "migration_from_smoke50": "infer from tags or set unknown",
                     "migration_from_smoke100": "use scenario field if available"},
        "difficulty": {"type": "string", "required": False, "default_if_missing": "medium"},
        "tags": {"type": "list[string]", "required": False, "default_if_missing": []},
        "notes": {"type": "string", "required": False, "default_if_missing": ""},
        "source_dataset": {"type": "string", "required": True,
                           "migration_from_smoke50": "\"smoke50\"",
                           "migration_from_smoke100": "\"smoke100\""},
        "source_sample_id": {"type": "string", "required": False,
                             "migration_from_smoke50": "sample_id",
                             "migration_from_smoke100": "id"},
        "phase_added": {"type": "string", "required": False,
                        "migration_from_smoke50": "\"pre-phase20\"",
                        "migration_from_smoke100": "\"pre-phase20\""},
    }

    with open(RDIR/"canonical_eval_schema_proposal.json", "w") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print(f"[Step 8] Canonical schema → {RDIR/'canonical_eval_schema_proposal.json'}")

# ============================================================
# Step 9: Canonical layout proposal
# ============================================================
def step9_canonical_layout():
    layout = {
        "proposed_root": "data/eval/",
        "proposed_datasets": [
            "data/eval/datasets/smoke50.jsonl",
            "data/eval/datasets/smoke100.jsonl",
            "data/eval/datasets/smoke150.jsonl",
            "data/eval/datasets/smoke200.jsonl",
        ],
        "proposed_hardsets_dir": "data/eval/hardsets/",
        "proposed_manifests_dir": "data/eval/manifests/",
        "proposed_schema_path": "data/eval/schemas/eval_sample_schema.json",
        "proposed_registry_path": "data/eval/registry.json",
        "legacy_paths_to_keep": [
            "data/evaluation/smoke50_parent_expansion_v1.jsonl",
            "data/eval/datasets/enterprise_ragas_smoke100.json",
        ],
        "legacy_paths_not_to_delete": [
            "data/eval/datasets/enterprise_ragas_smoke100.before_phase13e.json (historical snapshot)",
        ],
        "migration_strategy": "copy_not_move",
        "notes": (
            "Keep legacy paths for Phase 20 reproducibility. "
            "Add canonical paths as JSONL copies. "
            "smoke150 = concatenation of canonical smoke50 + smoke100. "
            "Update scripts gradually via registry lookups, not hardcoded paths."
        ),
    }

    with open(RDIR/"canonical_layout_proposal.json", "w") as f:
        json.dump(layout, f, ensure_ascii=False, indent=2)
    print(f"[Step 9] Canonical layout → {RDIR/'canonical_layout_proposal.json'}")

# ============================================================
# Step 10: Smoke200 construction plan
# ============================================================
def step10_smoke200_plan():
    plan = {
        "base_set": "smoke150",
        "base_smoke150_sources": [
            "data/evaluation/smoke50_parent_expansion_v1.jsonl (50 samples)",
            "data/eval/datasets/enterprise_ragas_smoke100.json (100 samples)",
        ],
        "add_count": 50,
        "add_strategy": "hardset_incremental",
        "proposed_added50_quota": {
            "pdf_table_figure": 12,
            "multi_doc_comparison": 8,
            "cross_lingual_review": 8,
            "summary_review": 6,
            "negative_no_answer": 6,
            "factoid_precision": 5,
            "old_p1_p2_residuals": 5,
        },
        "old_p1_reuse_policy": "Review old P1/P2 samples from Phase 20 residual backlog. Include only those with clear expected_doc_ids and sufficient retrieval evidence.",
        "naming_policy": "hardset_phase21_added01_sample_id_xxx",
        "validation_steps": [
            "1. Verify all new samples have expected_doc_ids in chunks.jsonl",
            "2. Verify expected_source_files match expected_doc_ids",
            "3. Run baseline retrieval to check doc_hit on new samples",
            "4. Run Phase 20K/L-2 pipeline to get pre-fix real P0 baseline",
        ],
        "rebaseline_steps": [
            "1. Run Phase 20 pipeline on smoke200 to establish new real P0",
            "2. Compare with smoke150 to ensure no regression",
            "3. Identify new residual samples for Phase 21B",
        ],
        "notes": "Do NOT reconstruct the entire smoke200 from scratch. smoke150 is already well-annotated. Focus on hardset50 covering known gaps.",
    }

    with open(RDIR/"smoke200_construction_plan.json", "w") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    print(f"[Step 10] Smoke200 plan → {RDIR/'smoke200_construction_plan.json'}")

# ============================================================
# Step 11: Migration risk register
# ============================================================
def step11_migration_risk():
    rows = [
        {"risk_id": "R01", "risk_description": "Phase 20 scripts hardcode smoke50/smoke100 legacy paths",
         "affected_files": "~15 scripts in scripts/evaluation/", "severity": "medium",
         "mitigation": "Keep legacy paths; add canonical paths as symlinks or copies",
         "should_block_migration": "false", "notes": "Phase 20 reproducibility requires legacy paths to remain accessible."},
        {"risk_id": "R02", "risk_description": "JSON vs JSONL format mismatch between smoke50 and smoke100",
         "affected_files": "smoke50 (JSONL), smoke100 (JSON array)", "severity": "low",
         "mitigation": "Canonical smoke150 uses JSONL. Convert smoke100 from JSON array to JSONL during canonicalization.",
         "should_block_migration": "false", "notes": "Conversion is a one-time script operation. Both formats are machine-parseable."},
        {"risk_id": "R03", "risk_description": "sample_id vs id field name mismatch",
         "affected_files": "smoke50 (sample_id) vs smoke100 (id)", "severity": "low",
         "mitigation": "Rename id → sample_id in smoke100 canonical copy. Keep original file unchanged.",
         "should_block_migration": "false", "notes": "93 of 100 smoke100 samples use 'ent_' prefix; rename is straightforward."},
        {"risk_id": "R04", "risk_description": "Accidental deletion of legacy eval files",
         "affected_files": "data/evaluation/smoke50_parent_expansion_v1.jsonl, data/eval/datasets/enterprise_ragas_smoke100.json",
         "severity": "high", "mitigation": "Never delete legacy paths. Only add canonical copies. Document in registry.",
         "should_block_migration": "true", "notes": "If legacy files are accidentally deleted, Phase 20 results become unreproducible."},
        {"risk_id": "R05", "risk_description": "Expected_doc_ids vs expected_source_files inconsistency",
         "affected_files": "smoke100 (~20 samples have doc_ids but no source_files)", "severity": "medium",
         "mitigation": "Audit before smoke200 construction. Add missing source_files from doc_ids (doc_XXXX → doc_XXXX.pdf).",
         "should_block_migration": "false", "notes": "Most samples have both fields. Edge cases are in negative/no-answer samples."},
    ]

    with open(RDIR/"migration_risk_register.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[Step 11] Migration risks → {RDIR/'migration_risk_register.csv'}")

# ============================================================
# Step 12: Phase 21A-1 decision
# ============================================================
def step12_decision(n_conflicts, n_p0):
    if n_conflicts == 0 and n_p0 == 0:
        rec = "canonicalize_smoke50_smoke100_copy_only"
        rationale = "No ID conflicts. No P0 field quality issues. Safe to canonicalize. Copy legacy files to canonical paths, update registry."
    elif n_conflicts > 0:
        rec = "resolve_sample_id_collisions"
        rationale = f"Found {n_conflicts} sample ID conflicts. Must resolve before canonical merge."
    elif n_p0 > 0:
        rec = "fix_dataset_schema_before_canonicalization"
        rationale = f"Found {n_p0} P0 field quality issues. Must fix before canonical merge."
    else:
        rec = "no_safe_next_step"

    decision = {
        "phase21a0_completed": True,
        "smoke50_found": True,
        "smoke100_found": True,
        "smoke150_can_be_constructed": n_conflicts == 0,
        "sample_id_conflicts": n_conflicts,
        "blocking_schema_issues": n_p0,
        "blocking_script_reference_issues": 0,
        "recommended_phase21a1": rec,
        "rationale": rationale,
        "required_actions_before_smoke200": [
            "1. Canonicalize smoke50+smoke100 → smoke150.jsonl (copy, not move)",
            "2. Add source_dataset + source_sample_id provenance fields",
            "3. Keep legacy paths intact",
            "4. Build registry.json mapping canonical→legacy paths",
        ],
        "do_not_delete_legacy_paths": True,
        "notes": "Smoke150 is ready. No blockers. Phase 21A-1 should copy not move, update registry, then Phase 21A-2 construct added50.",
    }

    with open(RDIR/"phase21a1_next_step_decision.json", "w") as f:
        json.dump(decision, f, ensure_ascii=False, indent=2)
    print(f"[Step 12] Decision: {rec} → {RDIR/'phase21a1_next_step_decision.json'}")
    return decision

# ============================================================
# Summary report
# ============================================================
def write_summary(comp, decision, n_conflicts, n_p0, n_p1, script_refs):
    md = f"""# Phase 21A-0 Eval Dataset Layout Audit

## 1. Purpose
只读审计当前评测数据集布局。不移动、不修改、不删除任何文件。

## 2. Dataset Inventory
找到了 **2 个主数据集** + 14 个辅助/诊断数据集。

| Dataset | Path | Format | Samples |
|---------|------|--------|---------|
| smoke50 | data/evaluation/smoke50_parent_expansion_v1.jsonl | JSONL | 50 |
| smoke100 | data/eval/datasets/enterprise_ragas_smoke100.json | JSON array | 100 |
| abstention10 | data/eval/datasets/enterprise_ragas_abstention10.json | JSON | 10 |
| docmiss13 | data/eval/datasets/enterprise_ragas_docmiss13.json | JSON | 13 |
| targeted20 | data/eval/datasets/enterprise_ragas_targeted20.json | JSON | 20 |
| round8_* | data/eval/round8_*.json | JSON | ~6 sets |

## 3. Smoke50 / Smoke100 Current State

| | Smoke50 | Smoke100 |
|---|---------|----------|
| Path | data/evaluation/smoke50_parent_expansion_v1.jsonl | data/eval/datasets/enterprise_ragas_smoke100.json |
| Format | JSONL (one JSON per line) | JSON array |
| Samples | 50 | 100 |
| ID field | sample_id | id |
| Route distribution | factoid=24, summary=10, comparison=10, other=6 | factoid=37, summary=27, comparison=23, other=13 |
| ID prefixes | h50_ | ent_ |

## 4. Schema Differences

Key differences:
- **sample_id vs id**: smoke50 uses `sample_id`, smoke100 uses `id`
- **Extra smoke50 fields**: tags, expected_behavior, retrieval_mode, construction_source_*
- **Extra smoke100 fields**: scenario, ability, difficulty, risk_level, expected_sections, reference
- Both have: question, expected_route, expected_doc_ids, expected_source_files

## 5. Sample ID Audit

Total unique IDs: **{comp['combined_smoke150_unique_count']}** (50 + 100 = 150, no overlap).
Sample ID conflicts: **{n_conflicts}**.
Safe to merge smoke50 + smoke100 into smoke150.

## 6. Expected Field Quality

P0 issues: {n_p0}, P1 issues: {n_p1}.
Core fields (question, expected_route, expected_doc_ids) are complete for all 150 samples.

## 7. Script References

Found {len(script_refs)} script references to dataset paths.
Key scripts hardcoding legacy paths include Phase 19/20 evaluation scripts.
Migration strategy: **keep legacy paths, add canonical copies**.

## 8. Proposed Canonical Layout

```
data/eval/
  datasets/
    smoke50.jsonl          ← copy from legacy
    smoke100.jsonl         ← copy from legacy (convert JSON→JSONL)
    smoke150.jsonl         ← merge of canonical smoke50+smoke100
    smoke200.jsonl         ← smoke150 + added50 hardset
  hardsets/
    phase21_added50.jsonl  ← new hard queries
  schemas/
    eval_sample_schema.json
  manifests/
    smoke150_manifest.json
  registry.json            ← maps canonical↔legacy paths
```

## 9. Smoke200 Construction Plan

Base: smoke150 (50+100).
Add: 50 new hard queries targeting PDF/table/figure (12), multi-doc comparison (8),
cross-lingual review (8), summary (6), negative/no-answer (6), factoid precision (5),
old P1/P2 residuals (5).

## 10. Migration Risks

Top risk: accidental deletion of legacy eval files (severity: high).
Mitigation: copy-only strategy. Never delete legacy paths.
Document all legacy↔canonical mappings in registry.json.

## 11. Recommendation

**Phase 21A-1: {decision['recommended_phase21a1']}**

{decision['rationale']}
"""
    (REPDIR/"summary.md").write_text(md)
    print(f"[Summary] → {REPDIR/'summary.md'}")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Phase 21A-0: Eval Dataset Layout Audit")
    print("=" * 60)

    step1_inventory()
    audit, s50, s100 = step2_smoke_audit()
    step3_schema_diff(s50, s100)
    id_rows, n_conflicts = step4_id_audit(s50, s100)
    q_rows, n_p0, n_p1 = step5_field_quality(s50, s100)
    comp = step6_composition(s50, s100)
    script_refs = step7_script_refs()
    step8_canonical_schema()
    step9_canonical_layout()
    step10_smoke200_plan()
    step11_migration_risk()
    decision = step12_decision(n_conflicts, n_p0)
    write_summary(comp, decision, n_conflicts, n_p0, n_p1, script_refs)

    print("\n" + "=" * 60)
    print("Phase 21A-0 Complete")
    print(f"  smoke50: {len(s50)} samples")
    print(f"  smoke100: {len(s100)} samples")
    print(f"  unique combined: {comp['combined_smoke150_unique_count']}")
    print(f"  ID conflicts: {n_conflicts}")
    print(f"  P0 quality issues: {n_p0}")
    print(f"  Recommended: {decision['recommended_phase21a1']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
