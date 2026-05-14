# Phase 5C-7 Preflight

- TableEnhancementConfig.enabled default: `False` (`false` expected).
- Settings.from_env current TABLE_ENHANCEMENT_ENABLED env present: `False`.
- build_round1_kb.py enable paths: `--table-enhancement`, `--table-enhancement-dry-run`, or explicit `TABLE_ENHANCEMENT_ENABLED=true` through Settings.from_env. No code default enables it.
- kb_builder default parsed input: `data/paper_round1/parsed_clean`; expected old parsed_clean: `./data/paper_round1/parsed_clean`.
- kb_builder default chunk dir: `data/paper_round1/chunks`; expected old chunks: `./data/paper_round1/chunks`.
- ON parsed input path from resolver: `data/paper_round1/parsed_clean_phase5c7_switch_on`; isolated output dir: `data/paper_round1/parsed_clean_phase5c7_switch_on`.
- DRY-RUN parsed input path from resolver: `data/paper_round1/parsed_clean`; dry-run keeps downstream path at old parsed_clean.
- DRY-RUN chunk dir from resolver: `data/paper_round1/chunks`; dry-run keeps downstream path at old chunks if full builder continues.
- run_table_enhancement creates `output_dir` only when `dry_run` is false, so dry-run does not write enhanced parsed JSON.
- preprocess_and_chunk keeps Chunk dataclass top-level fields unchanged; table metadata is copied only into `source_block_metadata`.

## Required Answers

- TableEnhancementConfig.enabled 默认是否 false: yes.
- build_round1_kb.py 是否只有显式参数才开启: CLI yes; env override `TABLE_ENHANCEMENT_ENABLED=true` can also explicitly enable through Settings.from_env.
- kb_builder 默认路径是否仍走旧 parsed_clean: yes.
- dry_run 是否不会写 enhanced parsed JSON: yes.
- on 模式是否使用隔离 output_dir: yes.
- 是否存在任何默认启用风险: no code default risk observed; only explicit env/CLI can enable.
- 是否存在覆盖主路径风险: no resolver collision for default suffix; guard checks parsed/chunk/jsonl/bm25 path collisions.
- preprocess_and_chunk.py 是否仍保持 chunks 主字段不变: yes; no `table_related`, `table_object`, or `figure_object` top-level Chunk fields.
