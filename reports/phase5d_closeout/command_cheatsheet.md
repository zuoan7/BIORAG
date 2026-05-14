# Phase 5D Command Cheatsheet

Dry-run:

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

Isolated cleanup:

```bash
python scripts/ingestion/cleanup_false_fragment_captions.py \
  --input_dir data/paper_round1/parsed_clean \
  --output_dir /tmp/biorag_phase5d3_caption_cleanup/parsed_clean \
  --signoff_csv reports/phase5d_caption_cleanup_signoff/signoff_decisions.csv \
  --protected_csv reports/phase5d_caption_cleanup_signoff/protected_caption_review.csv \
  --audit_dir reports/phase5d_caption_cleanup_experiment \
  --mode conservative
```

Generate isolated chunks:

```bash
python scripts/ingestion/preprocess_and_chunk.py \
  --input_dir /tmp/biorag_phase5d3_caption_cleanup/parsed_clean \
  --output_dir /tmp/biorag_phase5d3_caption_cleanup/chunks
```

Check closeout validation:

```bash
rg "phase5d4_pass|retrieval_smoke_pass|protected_violation_count|cleanup_default_off" \
  reports/phase5d_caption_cleanup_validation/summary.md
```
