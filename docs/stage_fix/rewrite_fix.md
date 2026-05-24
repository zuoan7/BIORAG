# Stage Fix Notes: Query Wiring and Rerank Score-Floor Misses

## Context

This document records the current diagnosis for v3 B0+rewrite `doc_hit_parent_miss` failures and the recommended repair direction.

Current scope is diagnostic and planning only. Do not implement same-doc expansion, neighbor expansion, score-floor changes, support selector changes, or generation changes before the query wiring fix is designed and verified.

Relevant audit artifacts:

- `results/evaluation/v3_gold_child_source_to_retrieval_audit_20260524_gold_child_source_to_retrieval_audit/gold_child_source_to_retrieval_summary.json`
- `results/evaluation/v3_rerank_score_floor_low_score_audit_20260524_rerank_score_floor_low_score_audit/rerank_score_floor_low_score_summary.json`
- `results/evaluation/v3_rewritten_query_rerank_replay_20260524_rewritten_query_rerank_replay/rewritten_query_rerank_replay_summary.json`
- `reports/evaluation/v3_gold_child_source_to_retrieval_audit_20260524_gold_child_source_to_retrieval_audit/report.md`
- `reports/evaluation/v3_rerank_score_floor_low_score_audit_20260524_rerank_score_floor_low_score_audit/report.md`
- `reports/evaluation/v3_rewritten_query_rerank_replay_20260524_rewritten_query_rerank_replay/report.md`

Audit scripts added for this investigation:

- `scripts/evaluation/audit_v3_gold_child_source_to_retrieval.py`
- `scripts/evaluation/audit_v3_rerank_score_floor_low_score.py`
- `scripts/evaluation/audit_v3_rewritten_query_rerank_replay.py`

## Problem

The system has a mixed Chinese/English query and corpus mismatch:

- User questions are often Chinese.
- Candidate evidence chunks are mostly English, with some Chinese.
- Main retrieval in B0+rewrite can use a rewritten English mirror query.
- Rerank was still effectively using the original Chinese question in the audited path.

This causes reranker scoring to prefer same-document wrong parents whose text has stronger surface overlap with the Chinese query or visible numeric/table/entity terms, while the gold parent is filtered by score floor.

Important distinction:

- `score_floor` is not the root cause for most affected samples.
- It is a symptom amplifier: once the top candidate receives a much higher rerank score, `top_score * rerank_score_floor_ratio` filters out the gold parent.

## Audit Findings

Gold child source-to-retrieval audit:

- Target samples: 35 `doc_hit_parent_miss`.
- All 35 have `source_status`, `index_status`, `retrieval_status`, and `loss_stage`.
- Largest root bucket: `rerank`, 19 samples.
- Largest single loss stage: `rerank_score_floor`, 12 samples.

Low-score audit for the 12 score-floor samples:

- `reranker_prefers_same_doc_wrong_parent`: 9 samples.
- `reranker_raw_semantic_score_low`: 2 samples.
- `relative_floor_too_aggressive_for_moderate_target`: 1 sample.
- 11/12 have a large raw reranker score gap against the top candidate.
- 10/12 have a top competitor from the same expected document but wrong parent.
- 8/12 are parent-only gold labels without child-level gold text.

Rewritten-query rerank replay:

Same rerank candidates were replayed with the rewritten query only.

- Target samples: 12.
- Gold parent raw score improved: 9.
- Gold parent pre-floor rank improved: 4.
- Gold parent passed post-floor: 7.
- Gold parent reached final top10: 6.

Final top10 rescued by rewritten-query rerank:

- `v3_pc_012`
- `v3_pc_040`
- `v3_pc_074`
- `v3_pc_078`
- `v3_pc_150`
- `v3_pc_160`

Samples not materially helped:

- `v3_ra_002`
- `v3_ra_016`
- `v3_ra_022`
- `v3_ra_029`

These likely need separate handling for candidate quality, gold granularity, or guarded rerank policy.

## Diagnosis

The immediate fix should not be "lower score floor globally".

The stronger diagnosis is:

1. Retrieval and rerank are using inconsistent query representations.
2. Retrieval benefits from the English mirror.
3. Rerank continues to score mostly English evidence using the original Chinese question.
4. The score floor then drops gold parents whose score would improve if rerank used the same rewritten query representation.

Therefore, the next repair should target query wiring before score-floor policy.

## Recommended Design

Introduce an explicit query bundle contract across retrieval and rerank:

- `original_query`: user-facing Chinese or mixed-language query.
- `rewritten_query`: English mirror query from query rewrite.
- `rerank_query`: the actual query string passed to reranker.
- Optional later fields: `query_entities`, `query_metrics`, `query_language`, `rewrite_source`.

Short-term query choice:

- For B0+rewrite enabled evaluation, rerank should use `rewritten_query` when rewrite succeeds.
- Debug output must record the actual `rerank_query`.
- The original query should still be kept for intent detection, route analysis, negative intent guard, and user traceability.

Do not fully convert the whole system to English:

- Original Chinese query remains the source of user intent.
- Chinese tokens/entities should remain available for BM25 or fallback.
- The rewritten query is a mirror used for English-heavy evidence matching, not a replacement for the user query.

Avoid default full bilingual double retrieval:

- Dense and rerank double-pass is expensive.
- Instead, use a single chosen query representation by default.
- Add selective fallback only when signal is weak.

Medium-term retrieval strategy:

- Dense retrieval: prefer English mirror when corpus evidence is English-heavy.
- BM25: use English mirror plus extracted entities/aliases; preserve Chinese biomedical terms if present.
- Original Chinese fallback: trigger only under low-signal conditions, not for every query.

Possible fallback triggers:

- Query contains CJK and top retrieval/rerank evidence has weak entity coverage.
- Rewritten query differs substantially from original query.
- Table/figure/numeric question but rewritten query loses table, figure, numeric, accession, or metric terms.
- Top candidates fail to include expected key entities or symbols.

## Proposed Implementation Plan

1. Add a query bundle data shape or equivalent stage-local structure.
   Verify: retrieval and rerank debug output exposes `original_query`, `rewritten_query`, and `rerank_query`.

2. Wire rerank to use the rewritten query in B0+rewrite mode.
   Verify: replay-rescued samples show rerank raw score improvements in an integration or replay test.

3. Keep route analysis based on the original query.
   Verify: no route regression for comparison, summary, negative, and factoid samples.

4. Add a focused regression test for one rescued sample.
   Suggested sample: `v3_pc_012` or `v3_pc_040`.
   Verify: gold parent is no longer dropped by score floor when rewritten rerank query is used.

5. Re-run the focused audit before changing score floor.
   Verify: the 12 score-floor samples are re-bucketed and the rescued count matches or exceeds replay expectations.

## Acceptance Criteria

Minimum acceptance for the query wiring fix:

- Rerank debug records the actual `rerank_query`.
- In B0+rewrite enabled path, rerank uses rewritten query when rewrite is available.
- The replay-rescued set has measurable improvement in live/debug output.
- No same-doc expansion, neighbor expansion, support selector, generation, or global score-floor policy is changed in the same patch.

Expected impact:

- Primary expected impact: up to 6 of the 12 `rerank_score_floor` samples can move into final top10 based on fixed-candidate replay.
- Additional partial impact: 1 more sample can pass post-floor but may still miss final top10.
- Remaining samples require separate audit; do not force them through by globally lowering score floor.

## Non-Goals

Do not include these in the first fix:

- Global `rerank_score_floor_ratio` reduction.
- Same-doc parent expansion.
- Neighbor expansion.
- Support selector retention changes.
- Generation or citation binding changes.
- Full bilingual double-pass retrieval for every query.

## Open Questions

- Should `rerank_query` be exactly `rewritten_query`, or a bilingual combined query with original and rewritten text?
- Should comparison subqueries be generated from original query, rewritten query, or both?
- Should BM25 use a compact bilingual term bundle instead of raw rewritten text?
- How should parent-only gold labels be evaluated when the model picks a same-document chunk that more directly answers the question?
