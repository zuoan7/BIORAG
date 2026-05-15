# v7-phase6A Baseline Guardrail Audit

Generated at: 2026-05-15

## Scope

This is the v7-phase6A-1 guardrail audit. It checks the baseline boundaries that
later Phase6A table/figure/OCR/VLM audits must preserve.

No code, tests, index rebuilds, Qwen calls, RAGAS calls, OCR runs, or VLM runs
were performed for this report.

## Guardrail Result

Status: pass for Phase6A planning.

The repository has a documented official clean baseline and a separate legacy
production reference. Later v7-phase6A work must keep those boundaries intact.

## Official Clean Baseline

| Item | Value |
|---|---|
| name | `phase5f_official_clean_baseline` |
| status | official clean retrieval baseline |
| dataset path | `reports/phase5f_eval_semantic_enhancement_v2/strict_main_eval_set_v2.jsonl` |
| dataset SHA256 | `39e817bf492fe6d40a784dc457b9ab566cb3061d13fef6cec0443b19d5ca09b3` |
| dataset count | `90` |
| chunks path | `data/baselines/phase5f_official_clean_baseline/chunks/chunks.jsonl` |
| chunk count | `15,802` |
| BM25 path | `data/baselines/phase5f_official_clean_baseline/bm25/bm25_index.json` |
| BM25 records | `15,802` |
| Milvus URI | `data/baselines/phase5f_official_clean_baseline/milvus/milvus_lite.db` |
| Milvus collection | `synbio_phase5f_official_clean_baseline` |
| Milvus rows | `15,802` |
| vector dimension | `1024` |
| table enhancement enabled | `false` |
| caption cleanup enabled | `false` |

Official retrieval-only metrics:

| Metric | Value |
|---|---:|
| doc_hit@10 | 95.6% |
| stable_block_hit@10 | 95.6% |
| stable_block_hit@20 | 95.6% |

## Dataset Denominator

The official clean-main denominator remains:

`reports/phase5f_eval_semantic_enhancement_v2/strict_main_eval_set_v2.jsonl`

with SHA256:

`39e817bf492fe6d40a784dc457b9ab566cb3061d13fef6cec0443b19d5ca09b3`

Known distribution:

| query_type | count |
|---|---:|
| `table_content` | 31 |
| `caption_level_table` | 9 |
| `figure_caption` | 20 |
| `normal_control` | 30 |

Phase6A may classify these samples, but it must not modify this denominator or
reuse a changed denominator under the same name.

## Legacy Production Reference

`current_default` / `synbio_papers` remains a legacy production reference.

It must not be described as:

- the official clean baseline;
- a replacement for `phase5f_official_clean_baseline`;
- the denominator for Phase6A table/figure audit decisions.

It may be mentioned only as a legacy production reference when that distinction
is explicit.

## Experimental Variants And Default-Off Capabilities

Phase 5C table enhancement:

- status: default-off capability or experimental ON variant;
- not official baseline behavior;
- must not be silently enabled for Phase6A.

Phase 5D caption cleanup:

- status: default-off capability or isolated experimental output;
- not official baseline behavior;
- must not write back to production `parsed_clean`.

Phase 5E section metadata repair:

- not implemented;
- remains backlog.

## Not Implemented In Phase 5

The following are not implemented baseline capabilities:

- structured table extraction;
- row/cell-level table eval;
- OCR;
- VLM image understanding;
- `table_object` schema;
- `figure_object` schema;
- generation/RAGAS evaluation tied to Phase 5 closeout.

v7-phase6A may audit the need for these capabilities. It must not claim they
already exist.

## Documentation Drift

The following entry points contain older baseline narratives:

- `README.md` references a Phase 20 convergence baseline.
- `docs/README.md` references older Phase 9 / Phase 11 baseline state.
- `results/phase-reports/phase7_*.md` contains historical Phase 7 smoke reports.

These sources should not override the Phase 5F official clean baseline for
v7-phase6A.

## Required Future Comparison Pins

Any later comparison that claims relation to the official clean baseline must
pin:

- dataset path and SHA256;
- chunks path and SHA256;
- BM25 path and SHA256;
- Milvus collection and row count;
- vector dimension;
- retrieval config;
- `table_enhancement_enabled=false`;
- `caption_cleanup_enabled=false`;
- whether it is retrieval-only, structured retrieval, or generation eval.

## Phase6A-1 Conclusion

v7-phase6A can proceed to table and figure coverage audits only under these
conditions:

- official clean baseline remains unchanged;
- all new outputs stay under the v7-phase6A audit report directory;
- classifications are audit labels, not production behavior;
- OCR/VLM/structured table/schema/object-relation work remains candidate-only
  until a later closeout gate recommends a separate implementation phase.
