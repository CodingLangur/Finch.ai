"""Embedding generators for semantic search and vector storage."""
import hashlib
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx

from ..config import AppConfig, config as default_config


class BaseEmbedder(ABC):
    """Abstract interface for generating vector embeddings."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate vector embedding for a single text string."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for multiple text strings."""
        pass

    def embed_text_sync(self, text: str) -> List[float]:
        """Synchronously generate embedding for a text string."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If already in an active event loop, run via new thread or task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.embed_text(text))
                return future.result()
        else:
            return asyncio.run(self.embed_text(text))


class OllamaEmbedder(BaseEmbedder):
    """Local embedding generator powered by Ollama (e.g., nomic-embed-text)."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dimension: int = 768,
        timeout: float = 30.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dim = dimension
        self.timeout = timeout

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed_text(self, text: str) -> List[float]:
        """Generate vector embedding for a single text via Ollama /api/embed."""
        batch = await self.embed_batch([text])
        if batch and len(batch) > 0:
            return batch[0]
        return [0.0] * self.dimension

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a batch of strings via Ollama."""
        if not texts:
            return []

        cleaned_texts = [t.strip() for t in texts]
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # 1. Attempt modern Ollama /api/embed endpoint
            try:
                resp = await client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": cleaned_texts},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    embeddings = data.get("embeddings", [])
                    if embeddings:
                        self._dim = len(embeddings[0])
                        return embeddings
            except Exception:
                pass

            # 2. Fallback to single-prompt /api/embeddings endpoint
            results: List[List[float]] = []
            for t in cleaned_texts:
                try:
                    resp = await client.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.model, "prompt": t},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    emb = data.get("embedding", [])
                    if emb:
                        self._dim = len(emb)
                        results.append(emb)
                    else:
                        results.append([0.0] * self.dimension)
                except Exception as e:
                    results.append([0.0] * self.dimension)
            return results

    def embed_text_sync(self, text: str) -> List[float]:
        """Synchronously generate embedding using httpx.Client."""
        cleaned = text.strip()
        with httpx.Client(timeout=self.timeout) as client:
            try:
                resp = client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": [cleaned]},
                )
                if resp.status_code == 200:
                    embs = resp.json().get("embeddings", [])
                    if embs:
                        return embs[0]
            except Exception:
                pass

            try:
                resp = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": cleaned},
                )
                if resp.status_code == 200:
                    return resp.json().get("embedding", [0.0] * self.dimension)
            except Exception:
                pass

        return [0.0] * self.dimension


class MockEmbedder(BaseEmbedder):
    """Deterministic embedder for unit tests and offline environments."""

    def __init__(self, dimension: int = 768):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_vector(self, text: str) -> List[float]:
        """Generate a deterministic normalized vector based on tokens."""
        vec = [0.0] * self._dim
        words = text.lower().split()
        if not words:
            return vec

        for word in words:
            # Distribute word hash across coordinates
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            for i in range(8):
                idx = (h + i * 97) % self._dim
                vec[idx] += 1.0

        # L2 Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    async def embed_text(self, text: str) -> List[float]:
        return self._hash_vector(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_vector(t) for t in texts]

    def embed_text_sync(self, text: str) -> List[float]:
        return self._hash_vector(text)


def get_embedder(config: Optional[AppConfig] = None) -> BaseEmbedder:
    """Factory creating configured embedder instance."""
    cfg = config or default_config
    if cfg.embedding_provider.lower() == "mock":
        return MockEmbedder(dimension=cfg.embedding_dim)

    return OllamaEmbedder(
        model=cfg.embedding_model,
        base_url=cfg.ollama_host,
        dimension=cfg.embedding_dim,
        timeout=cfg.request_timeout,
    )
