# Parent Expansion: Default Configuration

## Status

`parent_expansion_enabled` is **enabled by default** (Phase 12H, 2026-05-07).

## Explicit Disable

```bash
export RETRIEVAL_PARENT_EXPANSION_ENABLED=false
```

Or set `retrieval.parent_expansion_enabled = False` in Settings.

## Dependencies

| Component | Path | Requirement |
|-----------|------|-------------|
| chunks | `data/paper_round1/chunks/chunks.jsonl` | Must match Milvus collection (10610 rows) |
| parent_index | `data/paper_round1/chunks/parent_index.jsonl` | Must match chunks.jsonl (36862 parents) |
| Milvus | `runtime/vectorstores/milvus/papers.db` | Row count must equal chunks count |

## Fallback Behavior

If `parent_index.jsonl` is missing or `ParentStore` unavailable, parent expansion gracefully falls back to `reason="parent_index_missing"` or `reason="parent_store_unavailable"`. Pipeline does not crash.

## Expansion Modes

- **factoid**: chunk_window neighbor + section_path context (light, ~1-2 added)
- **summary**: section expansion (Abstract/Conclusion/Results) within same doc
- **comparison**: section_path cross-doc evidence balancing
- **table/figure/caption**: caption_context + page_context (with false_table_trigger guarding)
- **negative_trigger_robustness**: explicitly avoids caption_mode for expression/phenotype queries

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `parent_expansion_enabled` | True | Master switch |
| `parent_expansion_max_total` | 12 | Hard cap on total added chunks |
| `parent_expansion_per_seed_limit` | 2 | Max added per seed chunk |
| `parent_expansion_caption_enabled` | True | Enable caption context |
| `parent_expansion_page_enabled` | True | Enable page context |
| `parent_expansion_summary_sections_enabled` | True | Enable summary section expansion |

## Guarding Mechanisms

- `false_table_trigger_guarded`: Detects and blocks table-mode activation for non-table queries
- `primary_doc_local_context_gating`: Restricts cross-doc expansion for mixed-seed scenarios
- `caption_mode` / `comparison_mode`: Only activated when query analysis supports them

## Scope Limitations

- Text-level parent expansion only (chunk relationships from structured PDF parsing)
- No OCR-based table/figure detection
- No image embedding or visual object grounding
- No `table_figure_index` (not implemented)

## Monitoring Recommendations

- `P0_count` — should not increase vs parent_off baseline
- `zero_citation_count` — should not increase
- `min_citation_pass_rate` — should not decrease
- `final_context_count p90` — should stay below max_total
- `parent_added_count` — avg should stay below 2
- `latency_p95` — should stay acceptable (< +2000ms delta)
- `comparison` — check for over-expansion or single-doc dominance
- `table_figure_caption` — check for `false_table_trigger_guarded` count and actual degradation
- `negative_trigger_robustness` — verify no `caption_mode` misfire

## Rollback

To revert to parent_expansion disabled default:

```bash
# Option 1: Environment variable
export RETRIEVAL_PARENT_EXPANSION_ENABLED=false

# Option 2: Code change
# Revert config.py line 95: parent_expansion_enabled: bool = False
```
