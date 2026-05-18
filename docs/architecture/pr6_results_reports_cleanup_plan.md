# PR6 Results And Reports Cleanup Plan

Cleanup PR6 starts from `main` after PR5 deleted the PR4-quarantined unknown
scripts. PR6 should inventory generated results and reports before any deletion
or archive move.

## Assumptions

- The cleanup goal remains low-risk repository reduction and boundary clarity.
- Production behavior, `/v1/ask`, retrieval, BM25, dense, hybrid, rerank,
  generation v2, the old generation path, and Phase7 preview behavior are not
  changed.
- The 5 retained `scripts/ingestion` unknown files remain out of scope for PR6.
- PR6 works on tracked files first. Untracked or ignored local outputs may be
  reported separately but are not cleanup targets unless they are intentionally
  added to git.
- Accepted baselines, current eval datasets, current manifests, test fixtures,
  and Phase7 preview fixtures are protected until a proof pass says otherwise.

## Scope

Inventory these tracked artifact families:

- `reports/phase*`
- `reports/v7_phase*`
- `results/phase*`
- `results/v7_phase*`
- `data/experiments/v7_phase*`
- `data/eval/`
- `data/evaluation/`

Classify each tracked file or artifact directory as one of:

- `protected_current_reference`: referenced by README, docs, tests, configs,
  scripts, app, src, or current baseline registry.
- `protected_test_fixture`: read by tests or used as a fixture by a protected
  script.
- `protected_accepted_baseline`: part of the accepted Phase20/Phase21 baseline,
  registry, manifest, or closeout trail.
- `protected_phase7_preview`: required by Phase7/table preview tests, fixtures,
  schema prototypes, or dry-run guardrails.
- `archive_candidate`: generated historical output with no current references
  after exact path and module/text proof.
- `unknown`: cannot be proven safe within PR6.

## Strict Boundaries

Do not:

- delete or move files in PR6 unless the delete scope is tiny and fully proven;
- edit `app/`, `src/`, production configs, baseline registry, or accepted
  baseline artifacts;
- change scripts, including the 5 retained ingestion unknown files;
- run RAGAS, Qwen, embedding, rerank, retrieval evaluation, or index builds;
- promote Phase7 table evidence/citation to production;
- treat directory names alone as proof that an artifact is stale.

## Proof Pass

For each candidate, record:

- path and size;
- git tracked status;
- generating script if discoverable;
- exact path references from `README.md`, `docs`, `tests`, `configs`, `scripts`,
  `app`, and `src`;
- references from `configs/baseline_registry.yaml`;
- whether tests read it directly or indirectly;
- whether Phase7/table preview focused tests depend on it;
- proposed disposition and reason.

Generated cleanup inventory docs should not be used as blockers when checking
whether a file is currently referenced.

## Verification

Minimum PR6 verification:

```bash
pytest --collect-only -q
python -m compileall app src scripts tests
pytest tests/test_generation_v2.py -q
pytest tests/test_phase7_table_retrieval_wiring_preview.py tests/test_phase7l_table_rag_smoke.py tests/test_phase7m_sandbox_contract_hardening.py tests/test_phase7q_table_citation_schema_prototype.py tests/test_phase7q1_table_citation_mapper_dry_run.py tests/test_phase7r_table_index_production_proposal.py tests/test_phase7s_production_readiness_dry_run.py tests/test_phase7t_table_preview_scaffold.py tests/test_phase7u_table_preview_eval_smoke.py tests/test_phase7v_fast_type_aware_merge.py tests/test_phase7w_slim_mainchain_preview.py tests/test_phase7x_final_default_on_table_preview.py -q
```

If PR6 only adds inventory documentation, focused tests may be recorded as
"not rerun; no executable or artifact changes" only if the final diff proves it
is documentation-only.

## Deliverables

- A results/reports artifact inventory document.
- A candidate matrix with protected/archive/unknown disposition.
- A summary of protected current references and accepted baseline artifacts.
- A next-step recommendation: either quarantine a small proven stale artifact
  set in PR7, or keep the artifacts and close the cleanup thread.

## Recommended PR Split

- PR6: inventory and proof only.
- PR7: quarantine only the PR6-proven stale tracked artifacts, if any.
- Later delete PR: delete PR7-quarantined artifacts after collect and focused
  checks remain stable.

