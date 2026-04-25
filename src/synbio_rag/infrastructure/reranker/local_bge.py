from __future__ import annotations

from pathlib import Path

from FlagEmbedding import FlagReranker


class LocalBGEReranker:
    def __init__(
        self,
        model_path: str,
        use_fp16: bool = True,
        batch_size: int = 8,
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        model_dir = Path(model_path)
        weight_file = model_dir / "model.safetensors"
        if not model_dir.exists():
            raise FileNotFoundError(f"Reranker model path not found: {model_path}")
        if not weight_file.exists():
            raise FileNotFoundError(f"Reranker weight file not found: {weight_file}")
        self.model = FlagReranker(model_path, use_fp16=use_fp16)

    def score_pairs(self, pairs: list[list[str]]) -> list[float]:
        scores = self.model.compute_score(pairs, batch_size=self.batch_size)
        if isinstance(scores, float):
            return [scores]
        return [float(score) for score in scores]
