from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synbio_rag import Settings
from src.synbio_rag.infrastructure.reranker import LocalBGEReranker


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache
def get_reranker() -> LocalBGEReranker:
    settings = get_settings()
    return LocalBGEReranker(
        model_path=settings.reranker.model_path,
        use_fp16=settings.reranker.use_fp16,
        batch_size=settings.reranker.batch_size,
    )


app = FastAPI(title="synbio-reranker-service", version="0.1.0")


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1)
    documents: list[str] = Field(..., min_length=1)


@app.get("/healthz")
def healthz() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "model_path": settings.reranker.model_path,
    }


@app.post("/v1/rerank")
def rerank(request: RerankRequest) -> dict:
    reranker = get_reranker()
    scores = reranker.score_pairs([[request.query, doc] for doc in request.documents])
    items = [
        {"index": idx, "score": float(score)}
        for idx, score in enumerate(scores)
    ]
    return {"items": items}
