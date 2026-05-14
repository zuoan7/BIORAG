# Phase 5D-2 False / Fragment Caption Sign-off Summary

## Input Status
- missing Phase 5D-1 files: none

## Required Answers
1. total candidate count: 178.
2. safe_to_demote count: 48.
3. keep_as_caption count: 13.
4. needs_manual_pdf_check count: 2.
5. eval_only_noise count: 95.
6. uncertain count: 20.
7. pattern label distribution:
|candidate_rule|safe_to_demote|keep_as_caption|needs_manual_pdf_check|eval_only_noise|uncertain|
|---|---|---|---|---|---|
|article_plus_single_letter_fragment|4|0|0|0|0|
|broken_organism_or_abbreviation_prefix|39|0|2|0|0|
|number_only_caption|0|3|0|27|0|
|supplementary_reference_fragment|2|2|0|6|3|
|very_short_no_semantic_anchor|3|8|0|62|17|
8. high-confidence safe_to_demote ratio: 43/75 = 57.3%.
9. protected short caption review: {'uncertain': 1, 'correctly_protected': 59} across {'table_caption': 30, 'figure_caption': 30}; protected short captions are common and mostly real, so length-only cleanup is unsafe.
10. recommendation to enter Phase 5D-3: yes; use an experimental default-off path first.
11. implementable rules: article_plus_single_letter_fragment; broken organism/abbreviation prefix with strong nearby continuation evidence; exact page-header/footer fragments; supplementary fragments only with strong organism-continuation evidence.
12. rules not implementable now: broad number_only_caption cleanup; broad very_short_no_semantic_anchor cleanup; continued caption cleanup; any cleanup of approved eval targets, doc_0367 Figure 5, or semantic-anchor captions.
13. 5D-3 default enabled: no; default should remain off/no until dry-run and sign-off.
14. PDF manual check needed: yes for the explicit needs_manual_pdf_check rows and optional follow-up for uncertain rows, but not as a prerequisite for the narrow safe rules.

## Protected Review Conclusion
- Protected short captions are numerous and mostly meaningful in the sampled review.
- This confirms that caption length alone must not drive cleanup.
- 5D-3 protection should include semantic anchors, approved eval targets, doc_0367 Figure 5, descriptive body length, and continued-caption guards.

## Output Files
- signoff_decisions.csv
- protected_caption_review.csv
- cleanup_rule_proposal.md
- summary.md
