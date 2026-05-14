# Phase 5D Cleanup Usage

## Default State

Caption cleanup is default-off. Normal builds should continue to use production `data/paper_round1/parsed_clean` unless a caller explicitly runs the cleanup script into an isolated output directory.

Do not write cleanup output directly over production `parsed_clean`. The validated path is isolated output, audit, chunk smoke, and only then a separate decision about any formal rebuild.

## Inputs

The cleanup script uses:

- production parsed input, for example `data/paper_round1/parsed_clean`
- sign-off decisions: `reports/phase5d_caption_cleanup_signoff/signoff_decisions.csv`
- protected review: `reports/phase5d_caption_cleanup_signoff/protected_caption_review.csv`
- isolated output directory
- isolated audit directory

CLI entry point:

```bash
python scripts/ingestion/cleanup_false_fragment_captions.py \
  --input_dir data/paper_round1/parsed_clean \
  --output_dir /tmp/biorag_phase5d3_caption_cleanup/parsed_clean \
  --signoff_csv reports/phase5d_caption_cleanup_signoff/signoff_decisions.csv \
  --protected_csv reports/phase5d_caption_cleanup_signoff/protected_caption_review.csv \
  --audit_dir reports/phase5d_caption_cleanup_experiment \
  --mode conservative
```

## Dry Run

Use dry-run first. It still requires an output directory argument, but it should be an isolated throwaway path:

```bash
python scripts/ingestion/cleanup_false_fragment_captions.py \
  --input_dir data/paper_round1/parsed_clean \
  --output_dir /tmp/biorag_phase5d_dry_run/parsed_clean \
  --signoff_csv reports/phase5d_caption_cleanup_signoff/signoff_decisions.csv \
  --protected_csv reports/phase5d_caption_cleanup_signoff/protected_caption_review.csv \
  --audit_dir /tmp/biorag_phase5d_dry_run/audit \
  --dry_run \
  --mode conservative
```

Expected audit outputs include:

- `cleanup_audit.csv`
- `doc_level_stats.csv`
- `summary.md`
- `examples.md`
- `protected_caption_check.csv`
- `parsed_clean_diff_summary.md`

## Isolated Cleanup

Run cleanup only to an isolated output directory:

```bash
python scripts/ingestion/cleanup_false_fragment_captions.py \
  --input_dir data/paper_round1/parsed_clean \
  --output_dir /tmp/biorag_phase5d3_caption_cleanup/parsed_clean \
  --signoff_csv reports/phase5d_caption_cleanup_signoff/signoff_decisions.csv \
  --protected_csv reports/phase5d_caption_cleanup_signoff/protected_caption_review.csv \
  --audit_dir reports/phase5d_caption_cleanup_experiment \
  --mode conservative
```

The conservative mode only demotes signed-off `safe_to_demote` cases that pass rule guards. It does not delete blocks and should retain original text with cleanup metadata.

## Check Protected Violation

Read the cleanup summary:

```bash
rg "protected caption violations|approved eval target preserved|doc_0367 Figure 5 preserved" \
  reports/phase5d_caption_cleanup_experiment/summary.md
```

Expected Phase 5D values:

- protected caption violations: 0
- approved eval target preserved: True
- doc_0367 Figure 5 preserved: true

For Phase 5D-4 validation, also check:

```bash
rg "protected_violation_count|doc_0367_figure5_preserved|likely_false_positive_demote" \
  reports/phase5d_caption_cleanup_validation/summary.md
```

Expected values:

- protected_violation_count: 0
- doc_0367_figure5_preserved: true
- likely_false_positive_demote: 0

## Check doc_0367 Figure 5 Directly

The sentinel block is `doc_0367`, block `p6_b0059`. It should remain `figure_caption` in the cleanup output:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("/tmp/biorag_phase5d3_caption_cleanup/parsed_clean/doc_0367.json")
doc = json.loads(path.read_text())
for page in doc["pages"]:
    for block in page["blocks"]:
        if block.get("block_id") == "p6_b0059":
            print(block.get("type"))
            print(block.get("text", "")[:200])
PY
```

Expected first line:

```text
figure_caption
```

## Generate Cleanup Chunks

Generate chunks from the isolated cleanup output into an isolated chunks directory:

```bash
python scripts/ingestion/preprocess_and_chunk.py \
  --input_dir /tmp/biorag_phase5d3_caption_cleanup/parsed_clean \
  --output_dir /tmp/biorag_phase5d3_caption_cleanup/chunks
```

This writes:

- `/tmp/biorag_phase5d3_caption_cleanup/chunks/chunks.jsonl`
- `/tmp/biorag_phase5d3_caption_cleanup/chunks/failed_docs.log`

Do not use production chunk paths for this experiment unless there is an explicit rebuild plan.

## Validate Before Any Future Enablement

Before cleanup is used for any formal build:

1. Run dry-run.
2. Run isolated cleanup.
3. Check audit summary and protected violations.
4. Check doc_0367 Figure 5.
5. Generate isolated cleanup chunks.
6. Run static audit and retrieval smoke.
7. Only then decide whether a separate formal rebuild/eval plan is needed.

The reason is simple: real short captions are common. Writing cleanup output directly to production `parsed_clean` would make parser changes irreversible without a controlled rebuild, diff, and retrieval evaluation plan.
