from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


class BGEM3Embedder:
    def __init__(self, model_path: str, dim: int = 1024, use_fp16: bool = True):
        from FlagEmbedding import BGEM3FlagModel

        self.dim = dim
        self.model = BGEM3FlagModel(model_path, use_fp16=use_fp16)

    def encode(self, texts: list[str]) -> list[list[float]]:
        output = self.model.encode(texts, batch_size=16, max_length=512)
        return [vec.tolist() for vec in output["dense_vecs"]]
