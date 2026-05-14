# Phase 5C Follow-Up Backlog

## P1

Parser false / fragment caption cleanup:
- Why: Some caption-like fragments and parser false positives still create noisy table/figure caption boundaries.
- Why not Phase 5C: Phase 5C intentionally focused on preserving existing table-like paragraph text, not changing parser structure.
- When: Next cleanup phase after merge.

Section metadata cleanup:
- Why: Better section metadata improves chunk grouping, filtering, and downstream evaluation quality.
- Why not Phase 5C: It is a broader parsed_clean quality issue and could affect non-table retrieval.
- When: Next cleanup phase after parser caption cleanup or in parallel if scoped carefully.

Normal eval set quality improvement:
- Why: Cleaner normal_control questions improve confidence that table enhancements do not hurt ordinary retrieval.
- Why not Phase 5C: Current normal_control already showed no regression; improving the set is evaluation infrastructure work.
- When: Before the next large retrieval regression cycle.

Formal on/off rebuild regression after merge:
- Why: Merge changes should be validated once in the merged branch with isolated ON/OFF outputs.
- Why not Phase 5C: Phase 5C-7 already validated the switch pre-merge; the post-merge run belongs to release hygiene.
- When: Immediately after merge, before any production-like ON use.

## P2

Structured table extraction pilot:
- Why: Some questions require true row/cell structure rather than paragraph-level retrieval.
- Why not Phase 5C: Phase 5C explicitly avoided pdfplumber/camelot/tabula and structured table objects.
- When: Start as Phase 5D/6 after retrieval baseline is stable.

Table row/cell-level retrieval eval:
- Why: Needed to measure structured extraction value and avoid confusing it with paragraph-preservation gains.
- Why not Phase 5C: Current eval validates text retrieval, not row/cell understanding.
- When: Alongside or after a structured table extraction pilot.

Figure caption-image association audit:
- Why: Figure-caption retrieval stayed stable, but image association remains unvalidated.
- Why not Phase 5C: This phase did not include image understanding or image crop work.
- When: After table enhancement closeout, if figure-related failure modes become priority.

Image crop / OCR feasibility audit:
- Why: OCR may matter for scanned or image-only table/figure content outside the current parsed text path.
- Why not Phase 5C: Phase 5A/B found OCR candidate count 0/50 for the selected audit, so it was not the current main path.
- When: As a separate feasibility phase with dedicated data and tooling decisions.

## P3

`table_object` / `figure_object` schema design:
- Why: A structured object model may be needed for richer table/figure reasoning.
- Why not Phase 5C: Phase 5C deliberately preserved chunk schema and avoided new object fields.
- When: Only after structured extraction requirements and eval metrics are clear.

Parent-child index evaluation:
- Why: Parent-child indexing may improve evidence grouping once table/figure structures are stable.
- Why not Phase 5C: Current work isolated table-like text preservation and did not retune index architecture.
- When: After retrieval baseline and table/figure object decisions stabilize.

Generation / RAGAS after retrieval baseline stable:
- Why: Generation quality should be evaluated only after retrieval behavior is stable and interpretable.
- Why not Phase 5C: Phase 5C was retrieval and integration validation, not generation evaluation.
- When: After merge, formal on/off retrieval regression, and any P1 cleanup needed for the evaluation set.
