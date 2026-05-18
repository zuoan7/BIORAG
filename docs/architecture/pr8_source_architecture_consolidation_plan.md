# PR8 Source Architecture Consolidation Plan

PR8 removes the legacy generation branch and leaves `/v1/ask` on generation v2
only. This is an explicit behavior-support decision: `GENERATION_VERSION=old`
and `GenerationConfig.version` are no longer compatibility surfaces.

## Assumptions

- The production API remains `/v1/ask`.
- Generation v2 remains the only runtime answer path.
- Retrieval, BM25, dense, hybrid, rerank, parent expansion, query rewrite,
  original-CN fallback, confidence, session, audit, and Phase7 table preview
  behavior stay unchanged.
- Phase7 table preview remains preview/debug-only and must not become formal
  table citation support.
- RAGAS, Qwen, embedding, rerank evaluation, retrieval evaluation, index builds,
  and model downloads are out of scope.

## Removal Plan

1. Prove the current runtime still reaches old generation.
   - Check `pipeline.py` imports and the `generation.version != "v2"` branch.
   - Check tests that assert `GENERATION_VERSION=old`.
   - Check scripts and docs that mention old generation as supported.

2. Make `SynBioRAGPipeline.answer()` v2-only.
   - Remove old branch dispatch.
   - Remove old generator/context/external-tool construction.
   - Keep v2 response/debug shape stable.
   - Keep `neighbor_expansion.py` only as the v2 neighbor-audit index source.

3. Remove old-only source and tests.
   - Delete `application/generation_service.py`.
   - Delete `application/context_builder.py`.
   - Delete `application/legacy_generation_flow.py`.
   - Delete old external tool adapter code used only by the old branch.
   - Delete tests that only exercise old generation/Round8 policy.

4. Remove old config surface.
   - Remove `GenerationConfig.version`.
   - Stop parsing `GENERATION_VERSION`.
   - Remove old Round8 policy config and old external-tool config.

5. Update documentation and scripts.
   - Replace old-support language with v2-only language.
   - Remove `GENERATION_VERSION=v2` no-op settings from evaluation scripts.
   - Keep historical artifact/report docs unchanged when they describe past PRs.

## Verification

Required checks:

```bash
python -m py_compile src/synbio_rag/domain/config.py src/synbio_rag/application/pipeline.py
git diff --check
pytest tests/test_generation_v2.py tests/test_parent_expansion.py tests/test_generation_v2_config_profiles.py -q
pytest tests/test_phase7t_table_preview_scaffold.py tests/test_phase7w_slim_mainchain_preview.py tests/test_phase7x_final_default_on_table_preview.py -q
pytest tests/test_phase21a9c_query_rewrite_wiring.py tests/test_phase20l_original_cn_fallback.py tests/test_pipeline_summary_supplement.py -q
pytest --collect-only -q
```

Expected result:

- no old generation import path remains under `src/`;
- `Settings.from_env()` no longer accepts an old generation selector;
- `/v1/ask` still returns generation v2 responses;
- Phase7 preview formal citation guard remains intact.
