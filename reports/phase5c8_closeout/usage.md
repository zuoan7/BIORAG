# Phase 5C Table Enhancement Usage

## 1. Default Off

Table enhancement is off by default:

```python
TableEnhancementConfig.enabled = False
```

With default settings, the old flow is preserved. `kb_builder` reads the normal `data/paper_round1/parsed_clean`, writes the normal chunks path, and does not add `table_related` metadata.

## 2. How To Enable

Enable table enhancement explicitly for a round-1 build:

```bash
python scripts/ingestion/build_round1_kb.py \
  --table-enhancement \
  --table-enhancement-output-suffix table_enhanced_exp \
  --table-enhancement-audit-dir reports/table_enhancement/table_enhanced_exp
```

Expected isolated paths with the default config:

- enhanced parsed_clean: `data/paper_round1/parsed_clean_table_enhanced_exp`
- enhanced chunks: `data/paper_round1/chunks_table_enhanced_exp/chunks.jsonl`
- enhanced Milvus URI for local file URI: `runtime/vectorstores/milvus/papers_table_enhanced_exp.db`
- enhanced collection: `synbio_papers_table_enhanced_exp`
- audit dir: `reports/table_enhancement/table_enhanced_exp`

Disable explicitly:

```bash
python scripts/ingestion/build_round1_kb.py --no-table-enhancement
```

Dry-run through the formal builder:

```bash
python scripts/ingestion/build_round1_kb.py \
  --table-enhancement-dry-run \
  --table-enhancement-audit-dir reports/table_enhancement/dryrun_YYYYMMDD
```

Important: in the current implementation, `--table-enhancement-dry-run` means the table enhancement step writes audit but does not write enhanced parsed JSON. The formal `build_round1_kb.py` builder still continues the broader round-1 pipeline afterward, using baseline downstream paths. Do not run that command against main paths unless you intentionally want the full builder behavior.

For audit-only dry-run, use the wrapper around the integrated module:

```bash
python scripts/ingestion/enhance_table_like_paragraphs_pilot.py \
  --all_docs \
  --input_dir data/paper_round1/parsed_clean \
  --output_dir /tmp/biorag_table_enhancement_dryrun_output \
  --audit_dir reports/table_enhancement/dryrun_YYYYMMDD \
  --dry_run
```

For a selected-doc dry-run:

```bash
python scripts/ingestion/enhance_table_like_paragraphs_pilot.py \
  --selected_docs reports/phase5c7_switch_validation/selected_docs.csv \
  --input_dir data/paper_round1/parsed_clean \
  --output_dir /tmp/biorag_table_enhancement_dryrun_output \
  --audit_dir reports/table_enhancement/dryrun_selected \
  --dry_run
```

Confirm outputs by checking:

```bash
ls data/paper_round1/parsed_clean_table_enhanced_exp
ls reports/table_enhancement/table_enhanced_exp
```

If `--table-enhancement-audit-dir` is omitted, the audit root defaults to `reports/table_enhancement` and the builder appends a timestamped run directory.

## 3. Output Files

When enabled, the table enhancement step can produce:

- isolated enhanced parsed_clean JSON files
- `association_audit.csv`
- `doc_level_stats.csv`
- `summary.md`
- `false_positive_review.md`

The later build stages can produce:

- isolated chunks, usually `chunks_<suffix>/chunks.jsonl`
- optional isolated Milvus index, depending on builder settings
- optional isolated BM25 cache if a BM25 build step is run separately

## 4. Safety Gate

Before using enhanced outputs for any production-like index, inspect:

- `accepted_long_prose`
- suspicious docs
- association_count
- confidence distribution
- schema same / top-level field diff
- chunk growth and token growth
- Milvus VARCHAR risk, if building an index
- BGE tokenizer truncation delta, if building an index

Recommended gates:

- `accepted_long_prose = 0`
- schema unchanged
- no top-level chunk field additions
- no uncontrolled per-doc association growth
- no main path touched
- no material normal_control regression in retrieval-only validation

## 5. Recommended Workflow

Use a staged workflow:

1. Run dry-run audit first.
2. Review audit outputs, especially low-confidence and suspicious cases.
3. Run ON build to isolated paths with an explicit suffix.
4. Build chunks from the isolated enhanced parsed_clean.
5. Build an independent experimental index only if needed.
6. Run retrieval-only regression before any production-like use.
7. Do not directly overwrite the main parsed_clean, chunks, Milvus, or BM25 paths.
