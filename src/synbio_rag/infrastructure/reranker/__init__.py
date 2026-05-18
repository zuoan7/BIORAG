"""Reranker infrastructure adapters."""

__all__ = ["LocalBGEReranker"]


def __getattr__(name: str):
    if name == "LocalBGEReranker":
        from .local_bge import LocalBGEReranker

        return LocalBGEReranker
    raise AttributeError(name)
