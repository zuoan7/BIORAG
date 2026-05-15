# v7-phase6A Classification Rubric

Generated at: 2026-05-15

## Purpose

This rubric defines labels for later v7-phase6A table and figure sample audits.
It is designed for human-reviewable classification. It does not require running
new retrieval, OCR, VLM, Qwen, or RAGAS.

## Common Fields

Each classification row should include:

| Field | Meaning |
|---|---|
| `sample_id` | Eval sample identifier |
| `query_type` | Original sample query type |
| `classification` | One label from the relevant table or figure taxonomy |
| `confidence` | `high`, `medium`, or `low` |
| `evidence_basis` | What evidence supports the label |
| `requires_human_review` | `true` or `false` |
| `notes` | Short explanation and any caveat |

## Evidence Basis Values

Use one or more of:

- `baseline_manifest`
- `phase5_closeout_report`
- `phase5_eval_sample`
- `existing_per_sample_result`
- `existing_topk_artifact`
- `chunk_text_inspection`
- `parsed_clean_inspection`
- `existing_audit_report`
- `design_doc_context`
- `human_pdf_needed`

Do not use `new_retrieval_run`, `new_ocr_run`, `new_vlm_run`, `new_qwen_run`, or
`new_ragas_run` in v7-phase6A unless a later explicit scope change authorizes
those actions.

## Confidence Values

| Confidence | Definition |
|---|---|
| `high` | Existing text/artifact evidence is sufficient to classify without PDF review |
| `medium` | Existing evidence points strongly in one direction but has gaps |
| `low` | Existing evidence is ambiguous; human PDF review or later targeted audit is needed |

If confidence is `low`, `requires_human_review` should usually be `true`.

## Table Sample Taxonomy

Apply to `table_content` and `caption_level_table` separately. Do not aggregate
the two query types into one denominator without also reporting split counts.

| Classification | Meaning |
|---|---|
| `text_sufficient` | Existing text chunks contain enough table-related evidence to answer the question if retrieved and selected |
| `text_sufficient_but_existing_retrieval_gap` | Existing artifacts indicate the needed text exists, but the existing baseline result did not surface it reliably |
| `structure_needed` | The answer depends on row/column relations, inherited headers, units, table footnotes, or cell-level alignment that flat text is unlikely to preserve |
| `parser_loss_suspected` | The target evidence appears likely to exist in the PDF/table but is missing, truncated, or malformed in parsed/chunk text |
| `needs_manual_pdf_check` | Existing artifacts are insufficient to decide whether text, structure, parser loss, or label quality is the root cause |
| `bad_target` | The eval target appears invalid, under-specified, mislabeled, or not answerable from the cited source |

## Figure Sample Taxonomy

Apply to figure-related samples without assuming image understanding exists.

| Classification | Meaning |
|---|---|
| `caption_only_sufficient` | Figure caption alone contains enough evidence to answer the question |
| `caption_plus_text_sufficient` | Caption plus nearby body text or surrounding chunks likely contain enough evidence |
| `OCR_needed_candidate` | The missing evidence is likely text inside the figure image, labels, axes, or panels |
| `VLM_needed_candidate` | The missing evidence likely requires visual interpretation beyond OCR, such as pathways, panel relations, curves, diagrams, or spatial arrangement |
| `needs_manual_pdf_check` | Existing artifacts cannot distinguish caption/text sufficiency from OCR/VLM need |
| `bad_target` | The figure target appears invalid, under-specified, mislabeled, or not answerable from the cited source |

## Guardrails For Classification

- Prefer `needs_manual_pdf_check` over overclaiming `OCR_needed_candidate` or
  `VLM_needed_candidate`.
- Use `parser_loss_suspected` only when existing parsed/chunk artifacts show
  missing, truncated, or malformed evidence.
- Use `structure_needed` only when the question requires table structure, not
  merely because the source is a table.
- `text_sufficient_but_existing_retrieval_gap` must be based on existing
  artifacts, not a new retrieval run.
- OCR/VLM labels are candidate needs, not implemented capabilities.
- Phase 5C table enhancement and Phase 5D caption cleanup must remain
  default-off and should not be treated as baseline behavior.

## Manual Review Queue

Any row should enter the manual review queue if:

- the target evidence cannot be verified in existing text artifacts;
- the query asks for visual interpretation of a figure panel;
- the query asks for row/cell-level values that are not preserved in text;
- the sample appears to have a bad or ambiguous target;
- classification confidence is low.

The manual queue is an audit output, not a permission to modify assets.
