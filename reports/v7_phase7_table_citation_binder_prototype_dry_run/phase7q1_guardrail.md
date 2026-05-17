# Phase7Q-1 Guardrail

Phase7Q-1 is a table citation mapper prototype dry-run. It converts already-existing table candidate/debug artifacts into `TableEvidenceCitation` prototype objects or blocked records.

Boundaries:

- Do not modify `src/`, `configs/`, the ingestion pipeline, the current `Citation` dataclass, or production `CitationBinder`.
- Do not generate answers or formal production citations.
- Do not promote preview table units into production evidence.
- Keep `source_csv_path`, `source_pdf_crop_path`, and markdown cards in debug provenance only.
- Do not call Qwen, LLMs, RAGAS, OCR, VLM, embedding, reranker, Milvus, or official BM25.
- Route C remains backlog.

Allowed behavior is limited to read-only artifact inspection and new Phase7Q-1 reports, data fixtures, results, scripts, and tests.
