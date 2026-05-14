# Phase 5D Backlog

## P1

### Eval-only noise filtering in probe/eval layer

Why needed: Phase 5D found 95 `eval_only_noise` captions. They are poor eval/probe targets even when they should not be changed in parsed data.

Why not Phase 5D scope: Phase 5D intentionally avoided changing non-`safe_to_demote` captions and did not redesign eval target selection.

When to do: before the next caption-heavy retrieval/eval set refresh.

### Manual review for uncertain / needs_manual_pdf_check

Why needed: 20 `uncertain` and 2 `needs_manual_pdf_check` samples remain unresolved.

Why not Phase 5D scope: Phase 5D only accepted signed-off high-confidence demotions and avoided PDF/manual adjudication.

When to do: before expanding cleanup rules or using these samples as training/eval anchors.

### Section metadata cleanup

Why needed: Some caption and paragraph artifacts are related to section-path or header/footer contamination, not only block type.

Why not Phase 5D scope: Phase 5D targeted false/fragment caption block cleanup only and did not alter section metadata semantics.

When to do: after caption cleanup is stable and before broad retrieval eval rebaselining.

### Normal eval set quality improvement

Why needed: Caption noise affects risk slices, but normal paragraph controls also need stable, high-quality target mapping.

Why not Phase 5D scope: Phase 5D used lightweight smoke only and did not rebuild the main eval set.

When to do: before the next full retrieval benchmark or any production-facing evaluation claim.

## P2

### Broader caption cleanup rules, phase two

Why needed: Patterns such as `number_only_caption` and `very_short_no_semantic_anchor` still contain noise.

Why not Phase 5D scope: Real short captions are common; broad rules require fresh sign-off and stronger guards.

When to do: only after a new audit/sign-off cycle and a default-off experiment.

### Parser false caption upstream fix

Why needed: The best long-term fix is to prevent false captions from being produced upstream.

Why not Phase 5D scope: Phase 5D intentionally added a conservative cleanup capability, not parser architecture changes.

When to do: after closeout, if repeated false-caption patterns continue to appear in new corpora.

### Structured table extraction pilot

Why needed: Flattened tables still limit row/cell evidence quality.

Why not Phase 5D scope: Phase 5D explicitly excluded pdfplumber/camelot/tabula and structured table extraction.

When to do: after retrieval baseline is stable and there is a scoped pilot dataset.

### Figure caption-image association audit

Why needed: Preserving captions is not the same as linking captions to the correct figure image or panel.

Why not Phase 5D scope: Phase 5D did not add image parsing, OCR, or figure-object modeling.

When to do: before any figure_object schema design or image-aware retrieval work.

## P3

### table_object / figure_object schema design

Why needed: Future object-level retrieval may need explicit table/figure identities, captions, and parent-child links.

Why not Phase 5D scope: Phase 5D was limited to block cleanup and kept chunk schema unchanged.

When to do: after table/figure extraction requirements and evidence contracts are clear.

### OCR / image table feasibility

Why needed: Some table and figure evidence may only exist in images.

Why not Phase 5D scope: OCR was explicitly out of scope and would require separate accuracy and cost validation.

When to do: after structured extraction pilot defines gaps that text-only parsing cannot cover.

### Parent-child index evaluation

Why needed: Object-level or section-level evidence may benefit from parent-child retrieval.

Why not Phase 5D scope: Phase 5D did not rebuild indexes or change retrieval architecture.

When to do: after retrieval baseline and cleanup behavior are stable.

### Generation / RAGAS after retrieval baseline stable

Why needed: Generation and RAGAS are only meaningful once retrieval inputs and target mapping are stable.

Why not Phase 5D scope: Phase 5D only ran static audit and lightweight retrieval smoke, with no Qwen or RAGAS.

When to do: after retrieval regression gates pass and a formal eval dataset is approved.
