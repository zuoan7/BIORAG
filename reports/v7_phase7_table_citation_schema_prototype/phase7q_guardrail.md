# Phase7Q Guardrail

Phase7Q only designs and prototypes a typed table citation schema. It does not implement production citation binding.

Hard boundaries:

- Do not modify the current `Citation` dataclass.
- Do not modify `CitationBinder` production behavior.
- Do not modify `src/`, `configs/`, or the ingestion pipeline.
- Do not generate an answer.
- Do not generate formal production citations.
- Do not promote preview table units into production evidence.
- Do not put `source_csv_path` or `source_pdf_crop_path` into a formal citation source.
- Do not call Qwen, any LLM, RAGAS, OCR, or VLM.
- Do not access Milvus, query official BM25, run embeddings, run a reranker, or build a production table index.
- Route C remains backlog.

Allowed outputs are limited to schema prototype reports, structured prototype files, offline validation results, and prototype tests under the Phase7Q paths.
