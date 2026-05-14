# Phase 5C-7 Switch Validation Summary

1. default-off 是否验证通过: true.
2. OFF 模式是否无 table_related metadata: true.
3. ON 模式是否生成 isolated enhanced parsed_clean: true (`/tmp/biorag_phase5c7_switch_on/parsed_clean_table_enhanced`).
4. ON 模式是否生成 audit: true (`/tmp/biorag_phase5c7_switch_on/audit`).
5. DRY-RUN 是否只生成 audit、不写 enhanced parsed JSON: true.
6. chunks JSONL 顶层字段是否保持: true.
7. 是否触碰主路径: false.
8. 是否需要 full index rebuild: no.
9. 是否建议进入 Phase 5C-8 integration closeout: true.
10. 是否建议默认开启: no.

## Key Stats

- selected_docs: 8
- off chunk_count: 204
- on chunk_count: 209
- on association_count: 57
- on confidence_distribution: {'high': 2, 'medium': 43, 'low': 12}
- accepted_long_prose: 0
- off table_related_metadata_count: 0
- on table_related_metadata_count: 57
- top_level_field_diff: {'only_in_off': [], 'only_in_on': []}
- main_path_diff: {}

## Gate Result

- Phase 5C-7 gate passed before test execution: true.
- py_compile: passed.
- table enhancement integration tests: 12 passed, 2 warnings.
- key regression tests: 76 passed, 2 warnings.
- Phase 5C-7 gate passed including tests: true.
