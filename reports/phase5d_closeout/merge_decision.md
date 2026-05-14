# Phase 5D Merge Decision

## 1. Recommendation

The cleanup capability can be merged, but it must remain default-off. Merging the capability does not mean formally enabling it in production builds.

The merge should not overwrite production `parsed_clean`, should not rebuild main indexes, and should not change the default build path. The validated behavior is isolated cleanup output plus audit and smoke validation.

## 2. Merge Conditions

The following conditions were met for the Phase 5D closeout decision:

- `py_compile` passed for the new validation script.
- cleanup unit / CLI tests passed: 13 passed.
- key regression tests passed: 76 passed.
- Phase 5D-4 static audit passed.
- Phase 5D-4S retrieval smoke passed.
- protected violation = 0.
- doc_0367 Figure 5 preserved and retrievable.
- false_positive_demote = 0.
- chunks schema unchanged.
- chunks JSONL main fields unchanged.
- `Chunk` dataclass unchanged.
- cleanup remains default-off.
- cleanup writes to isolated output paths.

## 3. Not Recommended

Do not:

- enable cleanup by default;
- delete blocks;
- write directly back to production `parsed_clean`;
- claim that Phase 5D solves all false captions;
- claim structured table or figure understanding;
- introduce table_object or figure_object as part of this merge;
- use a full index rebuild as a merge prerequisite;
- overwrite main Milvus or BM25 artifacts;
- treat `eval_only_noise`, `uncertain`, or `needs_manual_pdf_check` as automatically cleaned.

## 4. Merge Checklist

Before merge:

- Review `git diff`.
- Confirm unrelated working-tree files are not included.
- Confirm `/tmp` outputs are not committed.
- Decide which reports should be committed.
- Confirm cleanup is default-off.
- Confirm no production `parsed_clean` or chunks files are modified.
- Confirm tests are passing.
- Confirm `reports/phase5d_closeout/usage.md` matches the current CLI.

After merge:

- Default build remains unchanged.
- Cleanup is only run explicitly.
- Use dry-run before isolated cleanup.
- Use isolated cleanup before any chunk generation.
- If cleanup is ever proposed for a formal build, create a separate rebuild and evaluation plan first.

## 5. Decision

Decision: accept Phase 5D as a default-off cleanup capability.

Scope of acceptance: conservative false / fragment caption cleanup only. This acceptance does not authorize production overwrite, default enablement, full index rebuild, Qwen generation evaluation, RAGAS, OCR, or structured object schema work.
