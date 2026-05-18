# PR7 Local Artifact Inventory

This inventory records the ignored/untracked local artifact audit for PR7. It is
descriptive only: PR7 does not delete, move, promote, or track any artifact.

## Scope

Audit target:

- ignored and untracked files reported by git;
- local artifact roots named in the PR7 plan;
- official baseline paths already protected by PR6.

Out of scope:

- code changes;
- config changes;
- tracked artifact deletion;
- RAGAS, Qwen, embedding, rerank, retrieval evaluation, index builds, or model
  downloads;
- converting ignored local artifacts into tracked files.

## Command Summary

Read-only checks run:

```bash
git status --short --ignored
git ls-files --others --exclude-standard
git ls-files --ignored --others --exclude-standard
git ls-files --ignored --others --exclude-standard -z
du -sh reports results data/eval data/evaluation data/experiments data/baselines logs runtime vector_db models 2>/dev/null
```

Results:

- Pre-document non-ignored untracked files: 0.
- Ignored local file entries: 6130.
- `git status --short --ignored` reports ignored local directories and files
  only.
- No cleanup candidate was proven safe for repository deletion.

Top-level ignored entry counts from the NUL-delimited git list:

| Top-level path | Ignored entries |
| --- | ---: |
| `data` | 3886 |
| `results` | 1103 |
| `reports` | 440 |
| `scripts` | 281 |
| `tests` | 172 |
| `archive` | 114 |
| `src` | 61 |
| `models` | 59 |
| `.pytest_cache` | 5 |
| `vector_db` | 2 |
| `app` | 2 |
| `.claude` | 1 |
| `.env` | 1 |
| `AGENTS.md` | 1 |
| `CLAUDE.md` | 1 |
| `runtime` | 1 |

Notable local sizes:

| Path | Size | Notes |
| --- | ---: | --- |
| `models/BAAI` | 6.5G | Local BGE model assets. |
| `data/paper_round1` | 2.6G | Local paper corpus and parsed KB assets. |
| `data/baselines/phase5f_official_clean_baseline` | 466M | Official clean baseline assets. |
| `runtime/vectorstores` | 162M | Local Milvus runtime state. |
| `data/experiments` | 56M | Phase experiment fixtures and generated data. |
| `results` | 26M | Historical generated phase outputs. |
| Python `__pycache__` directories | 12M | Local interpreter cache. |
| `scripts` | 8.7M | Mostly ignored bytecode under script cache dirs. |
| `data/patent_round1` | 4.3M | Local patent source documents. |
| `reports` | 3.5M | Historical generated phase reports. |
| `data/eval` | 3.1M | Evaluation datasets, manifests, and historical local sets. |

## Classification

| Class | Paths | Count | Disposition |
| --- | --- | ---: | --- |
| `protected_official_baseline` | `reports/phase5f_eval_semantic_enhancement_v2/`, `data/baselines/phase5f_official_clean_baseline/` | 22 | Keep. Official baseline inputs and assets. |
| `protected_phase_evidence` | `data/eval/`, `data/evaluation/`, `data/experiments/`, `reports/`, `results/`, excluding the official baseline files counted above | 1847 | Keep. Phase evidence, evaluation data, constructed datasets, and preview fixtures. |
| `unknown_keep` | `data/paper_round1/`, `data/patent_round1/`, `data/paper_round1_pdf_rename_map.*` | 3560 | Keep. Local corpus and source-document state referenced by scripts; not a cleanup deletion candidate. |
| `local_runtime_state` | `models/`, `runtime/`, `vector_db/`, `logs/`, `.env` | 63 | Keep unless the user explicitly asks for local machine cleanup. |
| `local_cache` | Python `__pycache__`, `.pyc`, and `.pytest_cache` entries | 635 | Optional local-only cleanup; not repository cleanup. |
| `unknown_keep` | `.claude/settings.local.json`, `AGENTS.md`, `CLAUDE.md` | 3 | Keep local workspace instructions/configuration. |

The counts are mutually exclusive and sum to 6130 ignored entries.

## Protected Official Baseline

`reports/phase5f_eval_semantic_enhancement_v2/` contains exactly:

- `strict_main_eval_set_v2.jsonl`
- `strict_main_eval_set_v2_summary.md`
- `summary.md`

The local dataset checksum matches the checksum pinned by the baseline assets:

```text
39e817bf492fe6d40a784dc457b9ab566cb3061d13fef6cec0443b19d5ca09b3
```

Reference checks found the official baseline in:

- `configs/baseline_registry.yaml`
- `docs/architecture/results_reports_artifact_inventory.md`
- `docs/architecture/pr7_local_artifact_hygiene_plan.md`
- `scripts/extraction/extract_tables_pdfplumber_v1.py`
- `scripts/evaluation/phase7l_table_rag_smoke_common.py`
- `scripts/evaluation/phase7m_contract_regression_check.py`
- Phase7 extraction/evaluation tests and scripts that guard against using
  official BM25/Milvus assets in preview-only paths.

Disposition: keep in place locally. Do not delete, rewrite, replace, or promote
this directory in cleanup.

## Phase Evidence And Generated Outputs

Ignored phase/evaluation artifacts account for 1850 entries before excluding
the three official baseline report files:

| Scope | Ignored entries | Size | Notes |
| --- | ---: | ---: | --- |
| `data/eval` | 36 | 3.1M | Historical evaluation datasets, hardsets, manifests, and RAGAS notes. |
| `data/evaluation` | 4 | 80K | Historical evaluation fixtures. |
| `data/experiments` | 267 | 56M | Phase6/Phase7 table extraction, review, QA, and preview data. |
| `reports` | 440 | 3.5M | Phase5/Phase6/Phase7 reports, including the official baseline report dir. |
| `results` | 1103 | 26M | Phase18-21 and Phase7 generated result packages. |

A broad reference scan across `README.md`, `docs`, `tests`, `configs`,
`scripts`, `app`, and `src` found many direct references to these families.
Examples include:

- `README.md` links to retained Phase7 report outputs and names local
  `models/` and `runtime/` directories.
- `docs/architecture/current_inventory.md` explicitly warns that
  `reports/phase*`, `results/phase*`, `data/paper_round1/`,
  `data/experiments/`, `models/`, and `runtime/` must not be deleted by
  directory name alone.
- Phase7 extraction scripts read and write many
  `data/experiments/v7_phase7_*`, `reports/v7_phase7_*`, and
  `results/v7_phase7_*` paths.
- Focused Phase7 tests read retained preview fixture paths.
- Evaluation scripts reference `data/eval/datasets/`, `results/phase*`, and
  model paths.

Disposition: keep. Some files are generated outputs, but PR7 did not prove any
package safe for repository cleanup. Package-specific deletion would need a
separate review against current scripts, tests, docs, and baseline contracts.

## Local Corpus

`data/paper_round1/` has 3555 ignored entries and is 2.6G on disk:

| Subpath | Ignored entries |
| --- | ---: |
| `data/paper_round1/parsed_raw` | 1416 |
| `data/paper_round1/parsed_preview` | 708 |
| `data/paper_round1/parsed_clean` | 708 |
| `data/paper_round1/paper` | 708 |
| `data/paper_round1/phase4_shadow_parsed_clean` | 11 |
| `data/paper_round1/chunks` | 4 |

`data/patent_round1/` has 3 ignored PDF entries and is 4.3M on disk.
`data/paper_round1_pdf_rename_map.csv` and
`data/paper_round1_pdf_rename_map.jsonl` are also ignored local files.

These paths are source corpus and KB construction state, not safe local cleanup
targets. Scripts under `scripts/ingestion/`, `scripts/extraction/`, and
`scripts/ops/` reference this corpus.

Disposition: keep as `unknown_keep`.

## Runtime, Models, And Cache

Local runtime/model state:

- `models/BAAI/bge-m3`: 4.3G
- `models/BAAI/bge-reranker-v2-m3`: 2.2G
- `runtime/vectorstores/milvus/papers.db`: 162M
- `vector_db/papers.db` and `.papers.db.lock`: local vector DB leftovers
- `logs/`: ignored runtime log directory
- `.env`: ignored local environment configuration

Disposition: keep unless the user explicitly asks for local machine cleanup.
These are not repository cleanup candidates.

Local cache:

- 30 Python `__pycache__` directories;
- 635 ignored cache entries including `.pyc` and `.pytest_cache`;
- 12M total Python bytecode cache size plus 60K `.pytest_cache`.

Disposition: optional local-only cleanup if requested. PR7 does not delete them
because the default action is inventory and classification.

## Recommendation

Close PR7 as documentation-only local artifact hygiene:

- no tracked deletion;
- no ignored artifact deletion by default;
- no baseline promotion;
- no change to `configs/baseline_registry.yaml`;
- no change to code or runtime behavior.

If the user wants local disk cleanup later, the narrow safe starting point is
cache-only deletion of Python `__pycache__`, `.pyc`, and `.pytest_cache`
entries. Model files, runtime vector stores, source corpus, official baselines,
and phase evidence should require explicit user approval and a separate local
cleanup command.
