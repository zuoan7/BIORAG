# Phase 5C Merge Decision

## 1. Recommendation

Recommend merging the Phase 5C table enhancement code with table enhancement default-off.

This is a code availability merge, not a production enablement decision. The feature should remain off unless explicitly enabled through config or CLI.

## 2. Merge Conditions

Merge conditions currently satisfied:

- tests passed
- switch validation passed
- chunk schema unchanged
- `Chunk` dataclass unchanged
- default off confirmed
- ON output paths isolated
- dry-run audit available
- audit outputs available
- rollback available by disabling the switch
- main parsed_clean/chunks/Milvus/BM25 were not overwritten during validation

## 3. Not Recommended

Do not:

- enable table enhancement by default
- directly overwrite the main parsed_clean, chunks, Milvus, or BM25 paths
- jump directly to Qwen or RAGAS evaluation
- claim structured table understanding
- claim OCR capability
- claim `table_object` or `figure_object` support
- claim row/cell-level table reasoning

## 4. Pre-Merge Checklist

- Review `git diff`.
- Confirm no unrelated files are included in the merge.
- Decide whether reports should be committed or left as local artifacts.
- Confirm `/tmp` outputs are not committed.
- Confirm tests still pass.
- Confirm `TableEnhancementConfig.enabled=False`.
- Confirm `build_round1_kb.py` only enables the feature through explicit CLI/config.
- Confirm usage notes are available to whoever will run the first ON build.

## 5. Post-Merge Checklist

- Confirm default build still runs with table enhancement off.
- For the first formal ON build, use an explicit isolated suffix.
- Preserve the audit directory from the ON build.
- If rebuilding an index, use a new collection and isolated BM25 path.
- If audit or safety gates fail, stop before index build.
- If behavior is problematic, turn the switch off and continue with the baseline path.
