"""Embeddings module for semantic and vector operations."""
from .embedder import (
    BaseEmbedder,
    MockEmbedder,
    OllamaEmbedder,
    get_embedder,
)

__all__ = [
    "BaseEmbedder",
    "OllamaEmbedder",
    "MockEmbedder",
    "get_embedder",
]
