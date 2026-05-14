# Phase 5D-3 Caption Cleanup Experiment Summary

- dry_run: false
- missing input reports: none
- total_docs: 708
- processed_docs: 708
- success_docs: 708
- failed_docs: 0
- cleanup_candidates_seen: 178
- demoted_count: 46
- skipped_eval_only_noise_count: 95
- skipped_uncertain_count: 20
- skipped_protected_count: 0
- skipped_manual_pdf_check_count: 2
- skipped_keep_as_caption_count: 13
- skipped_rule_guard_count: 2
- by_count: {'uncertain': 20, 'eval_only_noise': 95, 'keep_as_caption': 13, 'needs_manual_pdf_check': 2, 'safe_to_demote': 2}
- demoted_by_rule: {'phase5d3_broken_organism_or_abbreviation_prefix': 39, 'phase5d3_article_plus_single_letter_fragment': 4, 'phase5d3_page_header_footer_caption_fragment': 3}
- skipped_protected_rule counts: {'skip_rule_guard': 1, 'skip_descriptive_caption': 1}
- table_caption count before/after: 837 / 800
- figure_caption count before/after: 2898 / 2889
- protected caption violations: 0
- doc_0367 Figure 5 preserved: true
- approved eval target preserved: True
- schema compatibility status: compatible: parsed_clean pages/blocks structure preserved; block text preserved; no chunk schema fields changed
- recommendation for Phase 5D-4: Proceed to Phase 5D-4 static audit / retrieval smoke on the experimental output only; keep cleanup default-off.

## Doc-Level Stats Sample
|doc_id|source_file|candidates_seen|demoted_count|table_caption_before|table_caption_after|figure_caption_before|figure_caption_after|skipped_eval_only_noise|skipped_uncertain|skipped_manual_pdf_check|skipped_rule_guard|
|---|---|---|---|---|---|---|---|---|---|---|---|
|doc_0183|doc_0183.pdf|3|3|2|2|8|5|0|0|0|0|
|doc_0128|doc_0128.pdf|2|2|6|4|15|15|0|0|0|0|
|doc_0129|doc_0129.pdf|2|2|6|4|15|15|0|0|0|0|
|doc_0151|doc_0151.pdf|2|2|2|0|6|6|0|0|0|0|
|doc_0192|doc_0192.pdf|2|2|2|0|6|6|0|0|0|0|
|doc_0195|doc_0195.pdf|2|2|2|0|6|6|0|0|0|0|
|doc_0335|doc_0335.pdf|2|2|2|2|7|5|0|0|0|0|
|doc_0387|doc_0387.pdf|2|2|3|1|11|11|0|0|0|0|
|doc_0436|doc_0436.pdf|3|2|5|3|3|3|0|1|0|0|
|doc_0027|doc_0027.pdf|1|1|1|1|5|4|0|0|0|0|
|doc_0083|doc_0083.pdf|1|1|0|0|9|8|0|0|0|0|
|doc_0104|doc_0104.pdf|1|1|3|2|2|2|0|0|0|0|
|doc_0105|doc_0105.pdf|1|1|2|1|6|6|0|0|0|0|
|doc_0118|doc_0118.pdf|3|1|10|10|9|8|1|0|0|0|
|doc_0119|doc_0119.pdf|1|1|3|2|6|6|0|0|0|0|
|doc_0120|doc_0120.pdf|1|1|6|5|4|4|0|0|0|0|
|doc_0158|doc_0158.pdf|1|1|4|3|5|5|0|0|0|0|
|doc_0167|doc_0167.pdf|1|1|3|2|6|6|0|0|0|0|
|doc_0170|doc_0170.pdf|1|1|1|0|7|7|0|0|0|0|
|doc_0172|doc_0172.pdf|1|1|1|0|6|6|0|0|0|0|
|doc_0196|doc_0196.pdf|2|1|5|4|3|3|1|0|0|0|
|doc_0202|doc_0202.pdf|1|1|3|2|6|6|0|0|0|0|
|doc_0212|doc_0212.pdf|1|1|1|0|5|5|0|0|0|0|
|doc_0316|doc_0316.pdf|1|1|2|1|5|5|0|0|0|0|
|doc_0353|doc_0353.pdf|1|1|1|0|6|6|0|0|0|0|
|doc_0388|doc_0388.pdf|1|1|3|2|1|1|0|0|0|0|
|doc_0437|doc_0437.pdf|1|1|3|2|0|0|0|0|0|0|
|doc_0462|doc_0462.pdf|3|1|10|10|9|8|1|0|0|0|
|doc_0510|doc_0510.pdf|1|1|3|2|6|6|0|0|0|0|
|doc_0512|doc_0512.pdf|1|1|1|0|6|6|0|0|0|0|
|doc_0518|doc_0518.pdf|1|1|1|0|0|0|0|0|0|0|
|doc_0533|doc_0533.pdf|1|1|1|0|7|7|0|0|0|0|
|doc_0661|doc_0661.pdf|1|1|3|2|6|6|0|0|0|0|
|doc_0671|doc_0671.pdf|1|1|2|1|2|2|0|0|0|0|
|doc_0679|doc_0679.pdf|1|1|2|1|2|2|0|0|0|0|
|doc_0683|doc_0683.pdf|1|1|1|0|2|2|0|0|0|0|
|doc_0001|doc_0001.pdf|0|0|1|1|6|6|0|0|0|0|
|doc_0002|doc_0002.pdf|0|0|2|2|7|7|0|0|0|0|
|doc_0003|doc_0003.pdf|0|0|2|2|7|7|0|0|0|0|
|doc_0004|doc_0004.pdf|0|0|0|0|6|6|0|0|0|0|
