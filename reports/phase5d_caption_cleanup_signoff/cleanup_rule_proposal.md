# Phase 5D-3 Cleanup Rule Proposal

## Safe Rules
- article_plus_single_letter_fragment.
- broken_organism_or_abbreviation_prefix only when nearby text strongly shows the organism/abbreviation continuation, such as E. + coli, S. + cerevisiae, C. + jejuni, B. + breve, or a table-list context ending in 'listed/shown/summarized in'.
- very_short page-header/footer forms only with exact patterns such as 'FIGURE 3 5 of 12'.
- supplementary_reference_fragment only when it has the same strong organism-continuation evidence; in-text supplementary references should be filtered from eval targets first, not parsed-cleaned by default.

## Do-Not-Clean Rules
- short but meaningful captions with semantic anchors.
- captions containing Workflow, Overview, Strains, Plasmids, Primers, Pathway, Schematic, Model, Alignment, Time course, Summary, Comparison, Effect, Analysis, Assay, Expression, Production, Growth, Activity, or similar domain nouns.
- approved eval target captions and the doc_0367 Figure 5 sanity anchor.
- number-only captions when the adjacent block looks like a real caption body unless a future merger preserves both marker and title.
- continued captions such as 'Figure 3. Cont.' or 'Table 1 (continued)' by default.
- captions with enough descriptive content even if they are short.

## Recommended Action
- Do not delete blocks.
- Prefer demote/retype or metadata marking over destructive cleanup.
- Preserve the original text exactly.
- Record `original_block_type`.
- Record `caption_cleanup_rule_id`.
- Record `caption_cleanup_reason`.
- Keep downstream schema compatibility; do not change chunk JSONL main fields.
- Add an audit output listing every demoted block and rule id.
- Run 5D-3 on an experimental output path first; default should remain off/no until reviewed.

## Stop Conditions
- Stop if safe_to_demote count drops materially after implementation dry-run.
- Stop if any approved eval target or doc_0367 Figure 5 is touched.
- Stop if protected-caption review reveals broad semantic-anchor false positives.
- Stop if number-only or continued captions require PDF layout decisions to express safely.
- Stop if implementation requires OCR, PDF visual extraction, schema changes, or index rebuild as a prerequisite.
